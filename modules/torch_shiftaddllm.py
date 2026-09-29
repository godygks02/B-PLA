"""ShiftAddLLM (You et al., NeurIPS 2024) as a post-training baseline for GPT-2.

ShiftAddLLM reparameterizes every weight matrix of a pretrained LLM into
binary matrices with power-of-two scaling factors, so that ``W @ x`` becomes
shifts (activation times scale) plus table lookups and additions (the binary
matrices). No weight is trained; a GPTQ-style column sweep uses calibration
activations to decide the binary codes. It is the closest published method to
B-PLA on the multiplication side, which is why Table 6 needs its row.

The authors ship scripts for OPT, LLaMA, BLOOM, Mistral and Gemma but not
GPT-2, so this module is a port: the quantizer itself -- ``ShiftAddLLM`` in
``quant_methods/shiftaddllm.py`` and ``BCQuantizer`` -- is imported unchanged
from the authors' repository at a pinned commit, and only the layer walk
around it is ours. That walk reproduces ``opt_sequential`` of their
``model/opt.py``: every sublayer of block ``i`` sees the same calibration
inputs, produced by blocks ``0..i-1`` that are already quantized, and block
``i`` is quantized only after all of its Hessians are collected.

Two things in the upstream code decide how the rows may be labelled, and the
port records both rather than hiding them:

* ``--acc`` turns on QuIP-style incoherence processing: each weight is
  rescaled and rotated by random orthogonal butterflies before binarization
  and rotated back afterwards (``QuantMethod.postproc``). The weights the
  model is evaluated with are therefore dense floats, not binary codes in the
  original basis. Running them without multiplications needs the rotations
  applied to the activations, which the released code does not do.
* ``--lat`` has no incoherence processing. Its weights are binary codes with
  one set of scales per eight columns and an eighth of the rows, the layout
  the authors' LUT-GEMM kernel executes. This is the shift-and-add form.

``binary_code_check`` measures which of the two a saved weight actually is.

The attention products ``QK^T`` and ``PV`` and the vocabulary projection are
not weight matrices of the blocks, so ShiftAddLLM leaves them in floating
point. B-PLA's weighted scope converts ``QK^T`` and ``PV`` as well; the table
has to say so.
"""

from __future__ import annotations

import contextlib
import os
import random
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterator, Sequence

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

ROOT = Path(__file__).resolve().parents[1]

UPSTREAM_URL = "https://github.com/GATECH-EIC/ShiftAddLLM.git"
#: The commit every reported ShiftAddLLM number was produced with.
UPSTREAM_COMMIT = "1053837ae320392e6cf5611ab3378550a1f6ba11"
DEFAULT_UPSTREAM = ROOT / "third_party" / "ShiftAddLLM"

MODES = ("acc", "lat")
CHECKPOINT_FORMAT = "shiftaddllm-dequantized-v1"

#: WikiText-2 perplexities the paper reports for OPT-125M (Table 2), used to
#: check the port against the authors' own numbers before trusting the GPT-2
#: rows. Their evaluation: test split joined with blank lines, non-overlapping
#: windows of the model's full context.
PAPER_OPT125M_WIKITEXT2 = {
    "fp16": 27.65,
    ("acc", 3): 31.29,
    ("acc", 2): 51.15,
    ("lat", 3): 56.96,
    ("lat", 2): 712.55,
}


# ------------------------------------------------------------------ upstream


def upstream_root(root: str | os.PathLike | None = None) -> Path:
    return Path(root or os.environ.get("SHIFTADDLLM_ROOT") or DEFAULT_UPSTREAM)


def upstream_commit(root: Path) -> str | None:
    """The checked-out commit of the upstream clone, or None if not a clone."""

    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def import_upstream(root: str | os.PathLike | None = None):
    """Import the authors' quantizer classes unchanged; return (ShiftAddLLM, BCQuantizer).

    ``lut_gemm/quant.py`` imports the compiled ``lutgemm`` CUDA extension at
    module level, and ``quant_methods/shiftaddllm.py`` imports that file for
    its weight packers. Quantization never calls the extension -- it is the
    inference kernel -- so an empty stand-in module is registered instead of
    building it. Nothing else is patched.
    """

    root = upstream_root(root)
    if not (root / "quant_methods" / "shiftaddllm.py").is_file():
        raise FileNotFoundError(
            f"ShiftAddLLM sources not found under {root}. Run ./fetch_shiftaddllm.sh "
            "or point SHIFTADDLLM_ROOT at a clone of "
            f"{UPSTREAM_URL} at commit {UPSTREAM_COMMIT}."
        )
    sys.modules.setdefault("lutgemm", types.ModuleType("lutgemm"))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from quant_methods.shiftaddllm import ShiftAddLLM  # type: ignore[import-not-found]
    from quantizers.bcq_quant.quantizer import BCQuantizer  # type: ignore[import-not-found]

    return ShiftAddLLM, BCQuantizer


@contextlib.contextmanager
def cpu_fallback() -> Iterator[bool]:
    """Let the CUDA-only upstream quantizer run on a CPU, for smoke tests only.

    The upstream code calls ``tensor.cuda()``, ``torch.cuda.synchronize()`` and
    ``torch.cuda.empty_cache()`` unconditionally. On a machine without CUDA
    these become no-ops for the duration of the block, so the port can be
    exercised end to end on a laptop. With CUDA present nothing is touched.
    Results produced this way are marked ``cpu_smoke`` and are never reported.
    """

    if torch.cuda.is_available():
        yield False
        return
    saved = (torch.Tensor.cuda, torch.cuda.synchronize, torch.cuda.empty_cache)
    torch.Tensor.cuda = lambda self, *args, **kwargs: self  # type: ignore[method-assign]
    torch.cuda.synchronize = lambda *args, **kwargs: None  # type: ignore[assignment]
    torch.cuda.empty_cache = lambda *args, **kwargs: None  # type: ignore[assignment]
    try:
        yield True
    finally:
        torch.Tensor.cuda, torch.cuda.synchronize, torch.cuda.empty_cache = saved  # type: ignore[method-assign]


def upstream_args(
    mode: str,
    wbits: int,
    *,
    groupsize: int = -1,
    blocksize: int = 128,
    bcq_round: int = 50,
    apot_nums: int = 3,
    percdamp: float = 0.01,
) -> SimpleNamespace:
    """The argument namespace ``fasterquant`` reads, as ``parsers.parse_args`` builds it.

    The defaults are the upstream defaults and those of
    ``script/acc/eval_opt.sh`` (3 bits, full-row groups, 50 BCQ rounds); the
    per-mode switches are copied from the end of ``parse_args``.
    """

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}.")
    args = SimpleNamespace(
        wbits=wbits,
        groupsize=groupsize,
        blocksize=blocksize,
        bcq_round=bcq_round,
        apot_nums=apot_nums,
        percdamp=percdamp,
        acc=mode == "acc",
        lat=mode == "lat",
        gptq=False,
        lut_eval=False,
        quant_config=None,
        temp_storage=None,
        record_error=None,
        pre_gptqH=True,
        pre_rescale=False,
        pre_proj=False,
        pre_proj_extra=0,
        incoh_processing=False,
        act_order=True,
        use_bst=True,
        columnwise=mode == "acc",
        block_quant=mode == "lat",
    )
    if mode == "acc":
        args.incoh_processing = True
        args.pre_rescale = True
        args.pre_proj = True
    return args


# -------------------------------------------------------------------- models


def decoder_blocks(model: nn.Module) -> list[nn.Module]:
    """The repeated blocks of GPT-2 or OPT, in order."""

    transformer = getattr(model, "transformer", None)
    if transformer is not None and hasattr(transformer, "h"):
        return list(transformer.h)
    decoder = getattr(getattr(model, "model", None), "decoder", None)
    if decoder is not None and hasattr(decoder, "layers"):
        return list(decoder.layers)
    raise ValueError(f"Unsupported model {type(model).__name__}: expected GPT-2 or OPT.")


def _named_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """(full name, block) for every block, collected before anything is swapped."""

    block_ids = {id(block) for block in decoder_blocks(model)}
    return [(name, module) for name, module in model.named_modules() if id(module) in block_ids]


def find_layers(module: nn.Module, prefix: str = "") -> dict[str, nn.Module]:
    """``modelutils.find_layers`` with GPT-2's Conv1D added to the searched types."""

    if type(module) in (nn.Linear, Conv1D):
        return {prefix: module}
    found: dict[str, nn.Module] = {}
    for name, child in module.named_children():
        found.update(find_layers(child, f"{prefix}.{name}" if prefix else name))
    return found


def conv1d_to_linear(model: nn.Module) -> list[str]:
    """Swap every Conv1D inside the blocks for the equivalent nn.Linear, in place.

    Conv1D computes ``x @ W + b`` with ``W`` stored as [in, out]; nn.Linear
    stores [out, in]. The upstream ``fasterquant`` transposes Conv1D weights,
    but its QuIP pre/post-processing (``--acc``) does not, and fails on a
    non-square Conv1D such as GPT-2's fused ``c_attn``. Converting first makes
    the upstream code see exactly the layout it was written for. Returns the
    full names of the converted modules so the weights can be saved back in
    Conv1D layout.
    """

    converted: list[str] = []
    for block_name, block in _named_blocks(model):
        for name, module in list(find_layers(block).items()):
            if not isinstance(module, Conv1D):
                continue
            weight = module.weight.data  # [in, out]
            linear = nn.Linear(weight.shape[0], weight.shape[1], bias=module.bias is not None)
            linear = linear.to(device=weight.device, dtype=weight.dtype)
            linear.weight.data.copy_(weight.t())
            if module.bias is not None:
                linear.bias.data.copy_(module.bias.data)
            parent_name, _, attribute = name.rpartition(".")
            parent = block.get_submodule(parent_name) if parent_name else block
            setattr(parent, attribute, linear)
            converted.append(f"{block_name}.{name}")
    return converted


# --------------------------------------------------------------- calibration


def wikitext2_calibration(
    tokenizer,
    nsamples: int,
    seed: int,
    seqlen: int,
    dataset_id: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
) -> list[torch.Tensor]:
    """``datautils.get_wikitext2``: random windows of the *training* split.

    Same procedure as upstream -- the whole split joined with blank lines,
    ``random.seed(seed)`` then ``randint`` window starts -- with the dataset
    under its namespaced Hub id, which is where ``wikitext`` now lives.
    """

    from datasets import load_dataset

    train = load_dataset(dataset_id, dataset_config, split="train")
    encoded = tokenizer("\n\n".join(train["text"]), return_tensors="pt").input_ids
    rng = random.Random(seed)
    windows = []
    for _ in range(nsamples):
        start = rng.randint(0, encoded.shape[1] - seqlen - 1)
        windows.append(encoded[:, start : start + seqlen])
    return windows


@torch.no_grad()
def official_perplexity(
    model: nn.Module,
    tokenizer,
    seqlen: int,
    device: torch.device,
    dataset_id: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    max_windows: int | None = None,
) -> dict[str, float]:
    """WikiText-2 perplexity under the upstream ``opt_eval`` protocol.

    Test split joined with blank lines, non-overlapping windows of ``seqlen``
    tokens, each window scored without context from the previous one. This is
    the protocol behind the paper's numbers and is used only to check the port
    against them; Table 6 rows come from the common harness instead.
    """

    from datasets import load_dataset

    test = load_dataset(dataset_id, dataset_config, split="test")
    encoded = tokenizer("\n\n".join(test["text"]), return_tensors="pt").input_ids
    windows = encoded.numel() // seqlen
    if max_windows is not None:
        windows = min(windows, max_windows)
    loss_fn = nn.CrossEntropyLoss()
    total = 0.0
    for i in range(windows):
        batch = encoded[:, i * seqlen : (i + 1) * seqlen].to(device)
        logits = model(batch).logits.float()
        loss = loss_fn(logits[:, :-1, :].reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
        total += float(loss) * seqlen
    return {"perplexity": float(torch.exp(torch.tensor(total / (windows * seqlen)))),
            "windows": windows, "seqlen": seqlen}


# -------------------------------------------------------------- quantization


class _StopForward(Exception):
    """Raised after the block being calibrated, so later blocks never run."""


def _stop_after(_module, _inputs, _output):
    raise _StopForward


@torch.no_grad()
def quantize_model(
    model: nn.Module,
    calibration: Sequence[torch.Tensor],
    args: SimpleNamespace,
    shiftaddllm_cls,
    bcquantizer_cls,
    *,
    model_name: str,
    batch_size: int = 8,
    max_blocks: int | None = None,
    log: Callable[[str], None] = print,
) -> list[dict[str, object]]:
    """Quantize every weight matrix of every block, block by block, in place.

    Block ``i``'s Hessians come from a forward pass that stops right after
    block ``i``, with blocks ``0..i-1`` already quantized -- the inputs
    ``opt_sequential`` feeds it -- and every sublayer of block ``i`` is
    quantized only after all of them have been observed, as upstream does.
    Batching the calibration windows changes nothing: ``add_batch`` counts
    sequences and sums ``x x^T`` over every token either way.
    """

    device = next(model.parameters()).device
    use_cache = getattr(model.config, "use_cache", None)
    if use_cache is not None:
        model.config.use_cache = False

    blocks = decoder_blocks(model)
    if max_blocks is not None:
        blocks = blocks[:max_blocks]
    report: list[dict[str, object]] = []

    for index, block in enumerate(blocks):
        started = time.perf_counter()
        subset = find_layers(block)
        methods = {}
        for name, module in subset.items():
            method = shiftaddllm_cls(module)
            method.quantizer = bcquantizer_cls(
                module.weight.data.size(),
                groupsize=args.groupsize,
                wbits=args.wbits,
                rounds=args.bcq_round,
                use_bst=args.use_bst,
                apot_nums=args.apot_nums,
            )
            methods[name] = method

        def collector(method):
            def hook(_module, inputs, output):
                method.add_batch(inputs[0].data, output.data)
            return hook

        handles = [subset[name].register_forward_hook(collector(methods[name])) for name in subset]
        handles.append(block.register_forward_hook(_stop_after))
        try:
            for start in range(0, len(calibration), batch_size):
                batch = torch.cat(list(calibration[start : start + batch_size])).to(device)
                try:
                    model(batch)
                except _StopForward:
                    pass
        finally:
            for handle in handles:
                handle.remove()

        for name in subset:
            methods[name].post_batch()
        for name in subset:
            method = methods[name]
            method.preproc(
                preproc_gptqH=args.pre_gptqH,
                percdamp=args.percdamp,
                preproc_rescale=args.pre_rescale,
                preproc_proj=args.pre_proj,
                preproc_proj_extra=args.pre_proj_extra,
            )
            method.fasterquant(args, model_name=model_name, layer_name=f"{index}.{name}")
            method.free()
        seconds = time.perf_counter() - started
        report.append({"block": index, "sublayers": list(subset), "seconds": seconds})
        log(f"block {index}: quantized {len(subset)} matrices in {seconds:.1f} s")

    if use_cache is not None:
        model.config.use_cache = use_cache
    return report


# ----------------------------------------------------------------- analysis


def binary_code_check(weight_out_in: torch.Tensor, wbits: int, row_groups: int = 8) -> dict[str, object]:
    """Is this weight a ``wbits``-plane binary code in its own basis?

    A column of a binary-coded matrix, within one group of rows sharing a set
    of scales ``alpha_1..alpha_wbits``, can only take the values
    ``sum_j +-alpha_j`` -- at most ``2**wbits`` of them. ``--lat`` shares scales
    over an eighth of the rows and ``--acc`` (before its rotation is undone)
    over whole columns, so splitting every column into eight row groups is a
    test both layouts pass when the weight really is binary-coded. A rotated
    weight fails it by orders of magnitude. The test says nothing when a group
    has no more rows than ``2**wbits`` -- any matrix passes then -- which is
    what ``conclusive`` reports; every GPT-2 matrix has at least 96 rows per
    group.
    """

    weight = weight_out_in.detach().float().cpu()
    rows, columns = weight.shape
    if rows % row_groups:
        row_groups = 1
    limit = 2 ** wbits
    grouped = weight.reshape(row_groups, rows // row_groups, columns)
    ordered, _ = torch.sort(grouped, dim=1)
    distinct = 1 + (ordered[:, 1:, :] != ordered[:, :-1, :]).sum(dim=1)  # [row_groups, columns]
    worst = int(distinct.max())
    return {
        "conclusive": rows // row_groups > limit,
        "binary_coded": worst <= limit,
        "max_distinct_per_group": worst,
        "limit": limit,
        "groups_over_limit": int((distinct > limit).sum()),
        "groups": int(distinct.numel()),
    }


# ---------------------------------------------------------------- checkpoints


def quantized_weights(
    model: nn.Module,
    conv1d_names: Sequence[str],
    blocks: int | None = None,
) -> dict[str, torch.Tensor]:
    """The weight matrices of the first ``blocks`` blocks (all by default), in
    the layout of the original checkpoint."""

    conv1d = set(conv1d_names)
    weights: dict[str, torch.Tensor] = {}
    for block_name, block in _named_blocks(model)[:blocks]:
        for name, module in find_layers(block).items():
            full = f"{block_name}.{name}"
            tensor = module.weight.detach().float().cpu()
            weights[f"{full}.weight"] = tensor.t().contiguous() if full in conv1d else tensor.clone()
    return weights


def save_checkpoint(path: str | os.PathLike, weights: dict[str, torch.Tensor], meta: dict[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": CHECKPOINT_FORMAT, "meta": meta, "weights": weights}, path)


def load_checkpoint_into(
    model: nn.Module,
    path: str | os.PathLike,
    expected_model_id: str | None = None,
    allow_partial: bool = False,
) -> tuple[dict[str, object], int]:
    """Overwrite the block weights of a freshly loaded model; return (meta, count).

    Biases, embeddings, LayerNorms and the vocabulary projection are untouched,
    exactly as ShiftAddLLM leaves them. Refuses a checkpoint made for another
    model, and one that stopped early unless ``allow_partial`` (smoke tests).
    A CPU smoke checkpoint loads, but carries ``cpu_smoke`` into the result
    record, where the report refuses it.
    """

    blob = torch.load(path, map_location="cpu", weights_only=True)
    if blob.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint.")
    meta = dict(blob["meta"])
    if expected_model_id is not None and meta.get("model_id") != expected_model_id:
        raise ValueError(
            f"{path} was made for {meta.get('model_id')!r}, not {expected_model_id!r}."
        )
    if meta.get("partial") and not allow_partial:
        raise ValueError(f"{path} quantized only {meta.get('blocks_quantized')} blocks.")
    state = model.state_dict()
    count = 0
    for name, tensor in blob["weights"].items():
        if name not in state:
            raise KeyError(f"{path}: {name} is not a parameter of {type(model).__name__}.")
        if tuple(state[name].shape) != tuple(tensor.shape):
            raise ValueError(
                f"{path}: {name} has shape {tuple(tensor.shape)}, model expects "
                f"{tuple(state[name].shape)}."
            )
        state[name].copy_(tensor.to(dtype=state[name].dtype))
        count += 1
    return meta, count
