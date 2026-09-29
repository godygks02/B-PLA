"""
Training-free drop-in comparison of Exact / PAO / B-PLA on pretrained models.

Experiment A of ``2026-08-25 PAO 대 B-PLA 직접 비교 실험 설계``: insert the
forward primitives of Kosson and Jaggi (NeurIPS 2023) and of B-PLA into the
*same* pretrained checkpoint, with zero weight updates in every condition, and
measure how much of the exact model's behaviour survives.

Fairness rules enforced here:
  * one checkpoint, one sample list, one seed, one batching, shared across
    every backend; the exact model's logits are the reference for all of them;
  * replacement scopes are never mixed -- ``multiplication``, ``nonlinear`` and
    ``combined`` are separate runs;
  * B-PLA calibration is forward-only and is reported (sample count and time);
  * PAO gets no calibration because it has none to give, which is a property of
    the method and is stated as such rather than treated as a handicap;
  * wall-clock is not reported as a hardware result. Neither backend has native
    hardware support, so PyTorch timings measure emulation overhead only.

Results stream to JSON after each condition so a partial CPU run is still
usable.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.torch_bpla import (
    SharedBPLATables,
    TorchBPLAActivation,
    TorchBPLAConfig,
    calibrate_model_activation_range,
    replace_attention_matmuls,
    replace_gpt2_conv1d_and_gelu,
    replace_layer_norms,
    replace_linear_and_gelu,
)
from modules.torch_pao import (
    TorchPAOActivation,
    TorchPAOConfig,
    replace_pao_attention_matmuls,
    replace_pao_gpt2_conv1d_and_gelu,
    replace_pao_layer_norms,
    replace_pao_linear_and_gelu,
)
from modules.compute_energy import multiplier_cost_summary
from modules.torch_fqvit import replace_fqvit_layer_norms
from modules.torch_smoothquant import SmoothQuantConfig, smooth_model
from modules.torch_ptq import (
    TorchPTQConfig,
    calibrate_ptq_model,
    finalize_ptq_model,
    replace_ptq_attention_matmuls,
    replace_ptq_gpt2_conv1d,
    replace_ptq_linear,
)


#: Row-chunk size for the comparison metrics, in elements. 32M float64 values
#: is 256 MB per temporary, which keeps the whole comparison well under a
#: gigabyte for any vocabulary size.
_COMPARISON_CHUNK_ELEMENTS = 32_000_000

SCOPES = ("multiplication", "nonlinear", "combined")
BACKENDS = (
    "exact",
    "ptq-w8a8",
    "ptq-w8a8-static",
    "ptq-fqvit",
    "ptq-smoothquant",
    "ptq-smoothquant-static",
    "pao",
    "pao-alpha",
    "bpla-float",
    "bpla-dyadic",
    "chen-pam",
)

#: PAM and its single-constant error compensation are separate backends so one
#: matched run can carry both. The correction is cheap but not free -- it is a
#: second piecewise affine multiply, so it doubles PAM's integer additions --
#: and a table that showed only one of the two would either understate the
#: baseline or hide what the correction costs.
PAO_BACKENDS = {"pao", "pao-alpha"}

#: The two W8A8 recipes, kept as separate backends so a single matched run can
#: report both against the same reference. ``ptq-w8a8`` uses dynamic per-token
#: activation scales, which is what ZeroQuant, LLM.int8() and SmoothQuant's O1
#: setting all do and is the strong form of the baseline. ``ptq-w8a8-static``
#: uses static per-tensor scales from percentile calibration, the conventional
#: recipe. Reporting only the first would look like a straw man in reverse;
#: reporting only the second would be the straw man.
PTQ_GRANULARITY = {
    "ptq-w8a8": "token",
    "ptq-w8a8-static": "tensor",
    # SmoothQuant is a W8A8 row with the activation-outlier migration of Xiao et
    # al. (ICML 2023) applied first. Both granularities are offered because they
    # answer different questions and the method's own paper is about the second.
    #
    # ``ptq-smoothquant`` is the O1 setting: migration on top of dynamic
    # per-token scales. Per-token scaling already survives outliers on its own --
    # experiments/activation_outliers.json puts it at 92.11% agreement against
    # 68.28% for per-tensor min-max on GPT-2 -- so the migration has little left
    # to recover and the row mostly shows that.
    #
    # ``ptq-smoothquant-static`` is the setting SmoothQuant exists for: static
    # per-tensor activation scales, where a single outlier channel otherwise
    # ruins the scale for the whole tensor. Reporting only the first would
    # understate the method and leave the W8A8 rows open to the straw-man
    # charge; reporting only the second would hide that our per-token row was
    # already strong.
    "ptq-smoothquant": "token",
    "ptq-smoothquant-static": "tensor",
    # FQ-ViT quantizes activations layer-wise with a min-max observer, which is
    # our per-tensor setting with the percentile clipping turned off.
    "ptq-fqvit": "tensor",
}

#: The backends that need the SmoothQuant front-end before quantization.
SMOOTHQUANT_BACKENDS = {"ptq-smoothquant", "ptq-smoothquant-static"}

#: Backends whose weighted-scope product has an arithmetic cost model in
#: ``modules.compute_energy``: the tile-centre plane at level k (Chen-PAM), its
#: SPT form at (k, T, B), and the float-coefficient ceiling. Each result row
#: for these carries the per-product cost beside its fidelity so the
#: accuracy-cost trade can be read from the artifact alone.
COSTED_BACKENDS = {"chen-pam", "bpla-dyadic", "bpla-float"}

#: FQ-ViT (Lin et al., IJCAI 2022) is the only training-free method that reaches
#: the nonlinear operators, so it is the one baseline that has a combined-scope
#: row at all. It does not address GELU -- the paper never mentions it -- so its
#: coverage is int8 matmuls plus LayerNorm (Power-of-Two Factor) plus Softmax
#: (Log-Int-Softmax), with GELU left exact. That is narrower than B-PLA's
#: combined scope and the table has to say so, or a coverage difference reads as
#: a fidelity difference.
FQVIT_BACKEND = "ptq-fqvit"

#: W8A8 quantizes weights and activations and leaves the nonlinearities in
#: floating point, which is the convention the published numbers are measured
#: under. There is therefore no honest ``nonlinear`` or ``combined`` row to
#: report for it; quantizing GELU and Softmax too is a separate research line
#: (FQ-ViT, I-ViT) with its own baselines. Asking for one is refused rather than
#: silently answered with a multiplication-scope run under another label.
PTQ_SCOPES = ("multiplication",)


def _bpla_config(args: argparse.Namespace, affine_path: str) -> TorchBPLAConfig:
    return TorchBPLAConfig(
        prefix_bits=args.prefix_bits,
        affine_path=affine_path,
        dyadic_terms=args.dyadic_terms,
        nonlinear_dyadic_terms=args.nonlinear_dyadic_terms,
        max_shift=args.max_shift,
        mantissa_bits=args.mantissa_bits,
        activation_range=args.activation_range,
        linear_chunk_out=args.linear_chunk_out,
        anchor_constant=args.anchor_constant,
    )


def _replace_vit_intermediate_activations(module: nn.Module, factory) -> int:
    replaced = 0
    for child in module.modules():
        activation = getattr(child, "intermediate_act_fn", None)
        if activation is None or isinstance(activation, (TorchBPLAActivation, TorchPAOActivation)):
            continue
        child.intermediate_act_fn = factory()
        replaced += 1
    return replaced


def convert(
    model: nn.Module,
    backend: str,
    scope: str,
    args: argparse.Namespace,
    is_gpt2: bool,
) -> dict[str, object]:
    """Apply one backend at one scope and report exactly what was replaced."""

    replace_multiplication = scope in {"multiplication", "combined"}
    replace_nonlinear = scope in {"nonlinear", "combined"}
    record: dict[str, object] = {
        "linear_modules": 0,
        "activation_modules": 0,
        "attention_blocks": 0,
        "layernorm_modules": 0,
        "calibration_seconds": 0.0,
        "calibration_samples": 0,
    }
    if backend == "exact":
        return record

    if backend in PTQ_GRANULARITY:
        is_fqvit = backend == FQVIT_BACKEND
        allowed = ("multiplication", "combined") if is_fqvit else PTQ_SCOPES
        if scope not in allowed:
            raise ValueError(
                f"The {backend} backend has no {scope!r} scope. W8A8 leaves GELU, "
                "Softmax and LayerNorm in floating point by construction; FQ-ViT "
                f"reaches Softmax and LayerNorm but not GELU. Allowed: {list(allowed)}."
            )
        config = TorchPTQConfig(
            weight_bits=args.ptq_weight_bits,
            activation_bits=args.ptq_activation_bits,
            per_channel_weights=args.ptq_per_channel_weights,
            # FQ-ViT calibrates activations with min-max, not a percentile.
            activation_percentile=100.0 if is_fqvit else args.ptq_percentile,
            activation_granularity=PTQ_GRANULARITY[backend],
            softmax_log2_bits=args.fqvit_softmax_bits if is_fqvit else None,
        )
        if backend in SMOOTHQUANT_BACKENDS:
            # Migration runs on the still-exact model, so the channel profile is
            # the one the unquantized checkpoint produces, and it must happen
            # before the modules are swapped for their quantized forms.
            start = time.perf_counter()
            record.update(
                smooth_model(
                    model,
                    args.calibration_inputs,
                    lambda module, batch: module(**batch),
                    args.calibration_batches,
                    SmoothQuantConfig(alpha=args.smoothquant_alpha),
                    is_gpt2,
                )
            )
            record["smoothquant_seconds"] = time.perf_counter() - start
        if is_gpt2:
            record["linear_modules"] = replace_ptq_gpt2_conv1d(
                model, config, replace_lm_head=args.replace_lm_head
            )
        else:
            record["linear_modules"] = replace_ptq_linear(
                model, config, replace_conv2d=args.replace_conv2d
            )
        record["attention_blocks"] = replace_ptq_attention_matmuls(model, config, mode="ptq-full")
        if is_fqvit and replace_nonlinear:
            # GELU stays exact: activation_modules is left at zero on purpose,
            # and that zero is the coverage difference against B-PLA.
            record["layernorm_modules"] = replace_fqvit_layer_norms(model, config)
        start = time.perf_counter()
        # Same batches, same count, same forward-only contract as the B-PLA
        # calibration below, so neither method gets more data than the other.
        calibrate_ptq_model(
            model,
            args.calibration_inputs,
            lambda module, batch: module(**batch),
            args.calibration_batches,
        )
        record["ptq_summary"] = {
            **finalize_ptq_model(model),
            "activation_granularity": config.activation_granularity,
            "activation_percentile": (
                config.activation_percentile if config.activation_granularity == "tensor" else None
            ),
            "per_channel_weights": config.per_channel_weights,
            "weight_bits": config.weight_bits,
            "activation_bits": config.activation_bits,
        }
        record["calibration_seconds"] = time.perf_counter() - start
        record["calibration_samples"] = args.calibration_sample_count
        return record

    if backend in PAO_BACKENDS:
        config = TorchPAOConfig(
            matmul_chunk_out=args.linear_chunk_out,
            alpha=args.pao_alpha if backend == "pao-alpha" else None,
        )
        if is_gpt2:
            record["linear_modules"] = replace_pao_gpt2_conv1d_and_gelu(
                model,
                config,
                replace_multiplication,
                replace_nonlinear,
                replace_lm_head=replace_multiplication and args.replace_lm_head,
            )
        else:
            record["linear_modules"] = replace_pao_linear_and_gelu(
                model,
                config,
                replace_multiplication,
                replace_nonlinear,
                replace_conv2d=replace_multiplication and args.replace_conv2d,
            )
        if not is_gpt2 and replace_nonlinear:
            record["activation_modules"] = _replace_vit_intermediate_activations(
                model, lambda: TorchPAOActivation(config)
            )
        record["activation_modules"] = max(
            int(record["activation_modules"]),
            sum(1 for m in model.modules() if isinstance(m, TorchPAOActivation)),
        )
        if replace_multiplication:
            record["attention_blocks"] = replace_pao_attention_matmuls(
                model, config, mode="pao-full", approximate_softmax=replace_nonlinear
            )
        elif replace_nonlinear:
            record["attention_blocks"] = replace_pao_attention_matmuls(
                model, config, mode="exact", approximate_softmax=True
            )
        if replace_nonlinear and args.replace_layernorm:
            record["layernorm_modules"] = replace_pao_layer_norms(model, config)
        return record

    # Chen et al.'s PAM (IEEE TC 2022) is the exact-coefficient centre plane: their
    # Eq. (6) minimises the squared error over the tile and returns
    # [k0, k1, k2] = [-x_hat*y_hat, y_hat, x_hat], which is Eq. (10) of this paper.
    # PAM Level i partitions into 4^i sub-domains, so Level i corresponds to k = i.
    # The one thing that must be switched off is our power-of-two bypass, which
    # PAM's Eq. (24) has no counterpart for.
    is_chen_pam = backend == "chen-pam"
    config = _bpla_config(
        args, "float" if backend in {"bpla-float", "chen-pam"} else "dyadic"
    )
    if is_chen_pam:
        # No bypass (their Eq. (24) has none), but power-of-two *constants* are
        # shifts in any datapath, PAM's included. Routing the 2^-3 attention
        # scale through the approximate plane charged PAM for an error no PAM
        # deployment would incur; every other backend here already gets that
        # multiply exactly (Mitchell for PAO, the bypass for B-PLA).
        config = TorchBPLAConfig(**{**config.__dict__,
                                    "power_of_two_exact": False,
                                    "exact_power_of_two_constants": True})
        record["chen_pam_level"] = args.prefix_bits
    if args.exact_pot_constants and not config.exact_power_of_two_constants:
        config = TorchBPLAConfig(**{**config.__dict__, "exact_power_of_two_constants": True})
    record["exact_power_of_two_constants"] = bool(config.exact_power_of_two_constants)
    record["power_of_two_exact"] = bool(config.power_of_two_exact)
    record["anchor_constant"] = config.anchor_constant
    tables = SharedBPLATables(config)

    if replace_nonlinear and args.calibrate_activation:
        start = time.perf_counter()
        # Calibration runs on the *exact* model, before any replacement, and
        # observes only forward activations -- no labels, no weight updates.
        measured_range = calibrate_model_activation_range(
            model,
            args.calibration_inputs,
            lambda module, batch: module(**batch),
            args.calibration_batches,
        )
        record["calibration_seconds"] = time.perf_counter() - start
        record["calibration_samples"] = args.calibration_sample_count
        record["calibration_range"] = float(measured_range)
        config = TorchBPLAConfig(**{**config.__dict__, "activation_range": float(measured_range)})
        tables = SharedBPLATables(config)

    if is_gpt2:
        record["linear_modules"] = replace_gpt2_conv1d_and_gelu(
            model,
            config,
            replace_multiplication,
            replace_nonlinear,
            None,
            tables,
            replace_lm_head=replace_multiplication and args.replace_lm_head,
        )
    else:
        record["linear_modules"] = replace_linear_and_gelu(
            model,
            config,
            replace_multiplication,
            replace_nonlinear,
            None,
            tables,
            replace_conv2d=replace_multiplication and args.replace_conv2d,
        )
    if not is_gpt2 and replace_nonlinear:
        record["activation_modules"] = _replace_vit_intermediate_activations(
            model, lambda: TorchBPLAActivation("gelu", config, tables)
        )
    record["activation_modules"] = max(
        int(record["activation_modules"]),
        sum(1 for m in model.modules() if isinstance(m, TorchBPLAActivation)),
    )
    if replace_multiplication:
        record["attention_blocks"] = replace_attention_matmuls(
            model, config, tables, mode="bpla-full", approximate_softmax=replace_nonlinear
        )
    elif replace_nonlinear:
        record["attention_blocks"] = replace_attention_matmuls(
            model, config, tables, mode="exact", approximate_softmax=True
        )
    if replace_nonlinear and args.replace_layernorm:
        record["layernorm_modules"] = replace_layer_norms(model, config, tables)
    return record


# --------------------------------------------------------------------------- ViT


def build_vit(args: argparse.Namespace):
    from transformers import ViTForImageClassification

    model = ViTForImageClassification.from_pretrained(args.vit_model_id)
    model.eval()
    return model.to(args.device)


def prepare_vit_batches(args: argparse.Namespace) -> list[dict[str, torch.Tensor]]:
    from datasets import load_dataset
    from transformers import ViTImageProcessor

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    # Imagenette labels index its own ten classes, so they have to be lifted into
    # the checkpoint's 1000-way space. A full ImageNet-1k split already uses that
    # space; remapping it would silently score against the wrong classes, and the
    # top-1 would look plausible rather than obviously broken.
    remap_imagenette = "imagenette" in args.vit_dataset_id.lower()
    if remap_imagenette:
        from torch_bpla_vit_probe import IMAGENETTE_TO_IMAGENET
    else:
        IMAGENETTE_TO_IMAGENET = None

    processor = ViTImageProcessor.from_pretrained(args.vit_model_id)

    # Ask for one split by name. ``load_dataset(id)`` with no split materialises
    # every split, and for ImageNet-1k that is roughly 150 GB of training images
    # this experiment never scores. Only the held-out split is ever read.
    candidates = [args.vit_split] if args.vit_split else ["validation", "test"]
    split, failures = None, []
    for name in candidates:
        try:
            split = load_dataset(args.vit_dataset_id, split=name)
            break
        except Exception as error:  # unknown split, or the id has none by that name
            failures.append(f"{name}: {type(error).__name__}: {error}")
    if split is None:
        if args.vit_split:
            raise ValueError(
                f"Could not load split {args.vit_split!r} of {args.vit_dataset_id!r}. "
                + " | ".join(failures)
            )
        # No held-out split published: carve one out of train, as before.
        split = load_dataset(args.vit_dataset_id, split="train").train_test_split(
            test_size=0.3, seed=42
        )["test"]

    # A subset has to be drawn across the label space, not off the front of the
    # file. ImageNet-1k's validation split is stored in class order, so the first
    # 5,000 images are the first ~100 classes and a top-1 measured on them says
    # nothing about the checkpoint. Shuffling once with a fixed seed keeps the
    # draw reproducible and identical across backends.
    if args.vit_shuffle_seed is not None and args.num_samples < len(split):
        split = split.shuffle(seed=args.vit_shuffle_seed)

    def transform(batch):
        images = [image.convert("RGB") for image in batch["image"]]
        inputs = processor(images, return_tensors="pt")
        labels = [int(label) for label in batch["label"]]
        if remap_imagenette:
            labels = [IMAGENETTE_TO_IMAGENET[label] for label in labels]
        inputs["labels"] = torch.tensor(labels, dtype=torch.long)
        return inputs

    dataset = split.with_transform(transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    batches: list[dict[str, torch.Tensor]] = []
    seen = 0
    for batch in loader:
        batches.append({"pixel_values": batch["pixel_values"], "labels": batch["labels"]})
        seen += batch["labels"].numel()
        if seen >= args.num_samples:
            break
    return batches


def _batch_to(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move one batch, or return it untouched if it is already there.

    Batches are held in host memory when the evaluation set is large, so every
    runner moves what it is about to use. ``Tensor.to`` is a no-op when the
    tensor already lives on ``device``, which keeps the small-dataset path free.
    """

    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def run_vit(model: nn.Module, batches: list[dict[str, torch.Tensor]]) -> dict[str, object]:
    logits = []
    labels = []
    device = _model_device(model)
    # An emulated ViT pass over a full split runs for hours with nothing on the
    # log between the backend banner and its summary line, so a progress line
    # goes out every few batches: enough to tell a live job from a hung one and
    # to project its finish, rare enough not to matter.
    started = time.perf_counter()
    for index, batch in enumerate(batches, start=1):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        logits.append(model(pixel_values).logits.float().cpu())
        labels.append(batch["labels"].cpu())
        del pixel_values
        if index % 8 == 0 or index == len(batches):
            elapsed = time.perf_counter() - started
            print(
                f"    batch {index}/{len(batches)}  {elapsed/60:.1f} min elapsed, "
                f"~{elapsed / index * (len(batches) - index) / 60:.1f} min left",
                flush=True,
            )
    logits_all = torch.cat(logits)
    labels_all = torch.cat(labels)
    top5 = logits_all.topk(5, dim=-1).indices
    correct = top5.eq(labels_all.view(-1, 1).expand_as(top5))
    return {
        "logits": logits_all,
        "top1": 100.0 * correct[:, :1].sum().item() / labels_all.numel(),
        "top5": 100.0 * correct.sum().item() / labels_all.numel(),
        "samples": int(labels_all.numel()),
    }


# -------------------------------------------------------------------------- GPT-2


def build_gpt2(args: argparse.Namespace):
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(args.gpt2_model_id)
    model.eval()
    return model.to(args.device)


def prepare_gpt2_batches(args: argparse.Namespace) -> list[dict[str, torch.Tensor]]:
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained(args.gpt2_model_id)
    raw = load_dataset(args.gpt2_dataset_id, args.gpt2_dataset_config, split="test")
    text = "\n\n".join(line for line in raw["text"] if line.strip())
    encoded = tokenizer(text, return_tensors="pt").input_ids[0]

    window = args.gpt2_sequence_length
    batches: list[dict[str, torch.Tensor]] = []
    total = 0
    for start in range(0, encoded.numel() - window, window):
        chunk = encoded[start : start + window].unsqueeze(0)
        batches.append({"input_ids": chunk})
        total += window
        if total >= args.gpt2_target_tokens:
            break
    return batches


@torch.no_grad()
def run_gpt2(model: nn.Module, batches: list[dict[str, torch.Tensor]]) -> dict[str, object]:
    logits = []
    negative_log_likelihood = 0.0
    token_count = 0
    device = _model_device(model)
    for batch in batches:
        input_ids = _batch_to(batch, device)["input_ids"]
        output = model(input_ids).logits.float()
        logits.append(output.cpu())
        shift_logits = output[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        loss = nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="sum",
        )
        negative_log_likelihood += float(loss)
        token_count += int(shift_labels.numel())
    return {
        "logits": torch.cat([l.reshape(-1, l.size(-1)) for l in logits]),
        "perplexity": math.exp(negative_log_likelihood / token_count),
        "tokens": token_count,
        "samples": len(batches),
    }


# --------------------------------------------------------------------------- glue


class FidelityAccumulator:
    """Running fidelity statistics, fed either all at once or one batch at a time.

    Logit metrics are computed on *row-centered* logits. Softmax is invariant to
    a per-row constant, and for GPT-2 that constant carries 99.95% of the raw
    logit energy, so an uncentered gain or MAE mostly measures a shift the model
    never sees. The uncentered figures are still reported, clearly labelled, for
    comparison with work that quotes them.

    Every metric is a ratio of sums, so feeding rows in pieces gives exactly the
    same answer as one pass over the whole tensor. That is what makes streaming
    possible, and streaming is what decides whether a full-split run finishes at
    all: GPT-2 logits are tokens x vocabulary, so one float32 copy of the whole
    WikiText-2 test split is 54 GiB, and holding the reference beside it -- plus
    the ``torch.cat`` that materialises each one -- is what the OOM killer
    reacts to.
    """

    _KEYS = (
        "elements",
        "abs_difference",
        "squared_difference",
        "squared_reference",
        "cross",
        "uncentered_abs_difference",
        "uncentered_cross",
        "uncentered_squared_reference",
    )

    def __init__(self) -> None:
        self.totals = dict.fromkeys(self._KEYS, 0.0)
        self.agreements = 0
        self.rows = 0

    def update(self, current: torch.Tensor, reference: torch.Tensor) -> None:
        rows = current.shape[0]
        chunk = max(1, min(rows, _COMPARISON_CHUNK_ELEMENTS // max(1, current.shape[-1])))
        for start in range(0, rows, chunk):
            current_chunk = current[start : start + chunk].double()
            reference_chunk = reference[start : start + chunk].double()
            centered_current = current_chunk - current_chunk.mean(dim=-1, keepdim=True)
            centered_reference = reference_chunk - reference_chunk.mean(dim=-1, keepdim=True)
            centered_difference = centered_current - centered_reference

            self.totals["elements"] += float(current_chunk.numel())
            self.totals["abs_difference"] += float(centered_difference.abs().sum())
            self.totals["squared_difference"] += float(centered_difference.pow(2).sum())
            self.totals["squared_reference"] += float(centered_reference.pow(2).sum())
            self.totals["cross"] += float((centered_current * centered_reference).sum())
            self.totals["uncentered_abs_difference"] += float(
                (current_chunk - reference_chunk).abs().sum()
            )
            self.totals["uncentered_cross"] += float((current_chunk * reference_chunk).sum())
            self.totals["uncentered_squared_reference"] += float(reference_chunk.pow(2).sum())
            self.agreements += int(
                (current_chunk.argmax(dim=-1) == reference_chunk.argmax(dim=-1)).sum()
            )
        self.rows += rows

    def result(self) -> dict[str, float]:
        elements = self.totals["elements"]
        mean_squared_difference = self.totals["squared_difference"] / elements
        mean_squared_reference = self.totals["squared_reference"] / elements

        return {
            "logit_mae": self.totals["abs_difference"] / elements,
            "logit_rmse": math.sqrt(mean_squared_difference),
            "logit_nrmse": math.sqrt(mean_squared_difference) / math.sqrt(mean_squared_reference),
            "argmax_agreement": 100.0 * self.agreements / self.rows,
            # The model-level counterpart of the per-product gain: a systematic
            # contraction shows up here even when the argmax survives.
            "output_gain": self.totals["cross"] / self.totals["squared_reference"],
            "uncentered_logit_mae": self.totals["uncentered_abs_difference"] / elements,
            "uncentered_output_gain": (
                self.totals["uncentered_cross"] / self.totals["uncentered_squared_reference"]
            ),
        }


def compare_to_reference(current: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Compare approximate outputs to the exact ones, both held in full."""

    accumulator = FidelityAccumulator()
    accumulator.update(current, reference)
    return accumulator.result()


@torch.no_grad()
def run_gpt2_streaming(
    model: nn.Module,
    reference: nn.Module | None,
    batches: list[dict[str, torch.Tensor]],
) -> dict[str, object]:
    """Score GPT-2 and compare against the exact model without storing logits.

    The exact model is re-run alongside the approximate one so that each batch
    can be compared and discarded. That costs one extra exact forward pass per
    backend, which for GPT-2 is about 3 seconds over the whole WikiText-2 split
    against hours of emulation -- and it is the difference between a run that
    finishes and one the kernel kills. ``reference`` is None for the exact row,
    which needs no comparison.
    """

    accumulator = FidelityAccumulator()
    negative_log_likelihood = 0.0
    token_count = 0
    approximate_seconds = 0.0

    device = _model_device(model)
    for batch in batches:
        input_ids = _batch_to(batch, device)["input_ids"]

        started = time.perf_counter()
        output = model(input_ids).logits.float()
        if output.is_cuda:
            torch.cuda.synchronize()
        approximate_seconds += time.perf_counter() - started

        shift_logits = output[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        negative_log_likelihood += float(
            nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="sum",
            )
        )
        token_count += int(shift_labels.numel())

        if reference is not None:
            exact = reference(input_ids).logits.float()
            accumulator.update(
                output.reshape(-1, output.size(-1)),
                exact.reshape(-1, exact.size(-1)),
            )
            del exact
        del output

    outcome: dict[str, object] = {
        "perplexity": math.exp(negative_log_likelihood / token_count),
        "tokens": token_count,
        "samples": len(batches),
        "approximate_forward_seconds": approximate_seconds,
    }
    if reference is not None:
        outcome["streamed_metrics"] = accumulator.result()
    return outcome


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact / PAO / B-PLA training-free comparison.")
    parser.add_argument("--models", nargs="+", default=["vit"], choices=["vit", "gpt2"])
    parser.add_argument("--backends", nargs="+", default=list(BACKENDS), choices=BACKENDS)
    parser.add_argument("--scopes", nargs="+", default=["multiplication"], choices=SCOPES)

    parser.add_argument("--vit-model-id", default="google/vit-base-patch16-224")
    parser.add_argument("--vit-dataset-id", default="johnowhitaker/imagenette2-320")
    parser.add_argument(
        "--max-resident-batch-gb",
        type=float,
        default=4.0,
        help="Evaluation sets larger than this stay in host memory and are moved to "
             "the device one batch at a time. ImageNet-1k preprocesses to about 29 GB, "
             "which would otherwise sit on the device for the whole run.",
    )
    parser.add_argument(
        "--vit-shuffle-seed",
        type=int,
        default=None,
        help="Shuffle the ViT split with this seed before taking --num-samples. "
             "Required for any subset of ImageNet-1k, whose validation split is "
             "stored in class order: the first N images would otherwise cover only "
             "the first N/50 classes. Ignored when the whole split is used.",
    )
    parser.add_argument(
        "--vit-split",
        default=None,
        help="Dataset split to score. Default tries 'validation' then 'test'. "
             "Only this split is downloaded.",
    )
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)

    parser.add_argument("--gpt2-model-id", default="gpt2")
    parser.add_argument("--gpt2-dataset-id", default="Salesforce/wikitext")
    parser.add_argument("--gpt2-dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--gpt2-sequence-length", type=int, default=256)
    parser.add_argument("--gpt2-target-tokens", type=int, default=2048)

    parser.add_argument("--prefix-bits", type=int, default=4)
    parser.add_argument("--dyadic-terms", type=int, default=2)
    parser.add_argument(
        "--nonlinear-dyadic-terms",
        type=int,
        default=None,
        help="Term budget for the nonlinear tables; defaults to --dyadic-terms.",
    )
    parser.add_argument("--max-shift", type=int, default=16)
    parser.add_argument(
        "--mantissa-bits",
        type=int,
        default=None,
        help="Width of the B-PLA fixed-point mantissa datapath including the implicit "
             "leading one. Default keeps full float32 precision. The multiplier spends "
             "most of its energy on additions whose cost is linear in this width.",
    )
    parser.add_argument("--activation-range", type=float, default=4.0)
    parser.add_argument(
        "--anchor-constant",
        choices=("fixed", "spt"),
        default="fixed",
        help="Storage of each nonlinear segment's anchor constant. 'fixed' (default "
             "since 2026-09-11) keeps it on the --mantissa-bits datapath grid, which "
             "at full width is the exact constant: SPT then applies to the slope "
             "only. 'spt' quantizes it to --nonlinear-dyadic-terms signed "
             "power-of-two terms like the slope, which reproduces the runs before "
             "that date. It is only ever added, so 'fixed' removes no shift-add and "
             "changes only the nonlinear scopes.",
    )
    parser.add_argument(
        "--exact-pot-constants",
        dest="exact_pot_constants",
        action="store_true",
        help="Apply power-of-two constants (the 1/sqrt(64) attention scale) as "
             "shifts instead of routing them through the approximate multiplier. "
             "Always on for chen-pam; a no-op for the B-PLA paths, whose bypass "
             "already makes that product exact.",
    )
    parser.add_argument("--linear-chunk-out", type=int, default=128)
    parser.add_argument("--calibration-batches", type=int, default=2)
    parser.add_argument("--no-calibrate-activation", dest="calibrate_activation", action="store_false")
    parser.add_argument("--replace-layernorm", action="store_true")
    parser.add_argument(
        "--replace-lm-head",
        action="store_true",
        help="Also convert GPT-2's output projection (31%% of its weighted multiplies).",
    )
    parser.add_argument(
        "--replace-conv2d",
        action="store_true",
        help="Also convert ViT's patch-embedding convolution.",
    )
    parser.add_argument("--ptq-weight-bits", type=int, default=8)
    parser.add_argument(
        "--smoothquant-alpha",
        type=float,
        default=0.5,
        help="SmoothQuant migration strength (Xiao et al., ICML 2023). 0 leaves "
             "activations untouched, 1 moves the whole outlier into the weights; "
             "0.5 is the paper's default for OPT-family models.",
    )
    parser.add_argument(
        "--fqvit-softmax-bits",
        type=int,
        default=4,
        help="Log-Int-Softmax bit width for the ptq-fqvit backend. FQ-ViT uses 4.",
    )
    parser.add_argument("--ptq-activation-bits", type=int, default=8)
    parser.add_argument(
        "--ptq-percentile",
        type=float,
        default=99.99,
        help="Activation clipping percentile for ptq-w8a8-static. 100 disables clipping. "
             "Unused by ptq-w8a8, whose per-token scales are min-max over each row.",
    )
    parser.add_argument(
        "--ptq-per-tensor-weights",
        dest="ptq_per_channel_weights",
        action="store_false",
        help="Weaken PTQ to a single weight scale per tensor. Off: per-channel is standard.",
    )
    parser.add_argument(
        "--pao-alpha",
        type=float,
        default=1.056,
        help="PAO error-compensation constant (Sec. 2.7), used by the pao-alpha backend. "
             "The plain pao backend never applies it, as in the paper.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for both the model and the evaluation batches.",
    )
    parser.add_argument(
        "--stream-metrics",
        action="store_true",
        help="GPT-2 only: compare each batch against a second exact model and discard "
             "the logits, instead of holding the whole split twice in host memory. "
             "Required for the full WikiText-2 split, where one float32 copy of the "
             "logits is 54 GiB. Costs one extra exact forward pass per backend.",
    )
    parser.add_argument(
        "--save-logits",
        action="store_true",
        help="Cache raw logits per condition so metrics can be recomputed without rerunning.",
    )
    parser.add_argument(
        "--max-logit-cache-gb",
        type=float,
        default=2.0,
        help="Skip the logit cache above this size. Language-model logits are "
             "tokens x vocabulary and grow far faster than the runtime they save.",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "pao_vs_bpla_model.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    record: dict[str, object] = {
        "configuration": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "notes": [
            "Zero weight updates in every condition; PAO and B-PLA are both drop-in.",
            "PAO forward primitives follow Kosson and Jaggi (NeurIPS 2023) Eq. (5)-(20); "
            "verified in tests/test_pao.py against the Mogami int-addition trick.",
            "PAO's GELU is our composition from PA primitives; the paper's models use ReLU.",
            "Both W8A8 backends use per-output-channel symmetric weight quantization, "
            "simulated by quantize-dequantize with a float32 accumulator, which is the "
            "same accumulator every other backend uses.",
            "ptq-w8a8 uses dynamic per-token activation scales (ZeroQuant / LLM.int8() "
            "style); ptq-w8a8-static uses static per-tensor scales from percentile "
            "calibration. The static form is the conventional recipe and the dynamic "
            "form is the stronger one; both are reported.",
            "The W8A8 backends run at the multiplication scope only: W8A8 leaves the "
            "nonlinear paths in floating point by convention.",
            "Wall-clock is not reported: neither backend has native hardware support.",
            "Logit metrics are row-centered: softmax ignores a per-row constant, and for "
            "GPT-2 that constant holds 99.95% of the raw logit energy.",
            "multiplier_cost is the 45 nm arithmetic proxy of modules/compute_energy.py "
            "for one weighted product: fixed-point additions at the datapath width, "
            "shifts at zero, no table, register or memory traffic. Chen-PAM at level k "
            "is costed from Eq. (22) of Chen et al. (ICCAD 2020); B-PLA from the "
            "separable evaluator nu*m1 + mu*(m2 - nu) with T terms per coefficient.",
        ],
        "results": [],
    }
    results: list[dict[str, object]] = record["results"]  # type: ignore[assignment]

    # Fail before the first checkpoint is downloaded rather than hours into a
    # run: an unsupported combination costs the whole queue if it surfaces late.
    plain_w8a8 = [b for b in args.backends if b in PTQ_GRANULARITY and b != FQVIT_BACKEND]
    if plain_w8a8:
        unsupported = [scope for scope in args.scopes if scope not in PTQ_SCOPES]
        if unsupported:
            raise SystemExit(
                f"The W8A8 backends have no {unsupported} scope; W8A8 leaves the nonlinear "
                "paths in floating point. Run them with --scopes multiplication."
            )

    for model_name in args.models:
        is_gpt2 = model_name == "gpt2"
        if is_gpt2:
            batches = prepare_gpt2_batches(args)
            builder, runner = build_gpt2, run_gpt2
        else:
            batches = prepare_vit_batches(args)
            builder, runner = build_vit, run_vit
        # The evaluation set stays in host memory and moves one batch at a time.
        # Pinning it all to the device costs 29 GB of preprocessed tensors at
        # ImageNet-1k scale -- resident for the whole run, across every backend --
        # which leaves nothing for the model. The per-batch transfer is a few
        # seconds of PCIe against hours of emulation.
        resident = sum(
            value.numel() * value.element_size()
            for batch in batches
            for value in batch.values()
        )
        if resident > args.max_resident_batch_gb * 1024 ** 3:
            print(
                f"[{model_name}] evaluation set is {resident / 1024 ** 3:.1f} GiB; "
                f"streaming batches to {args.device} one at a time",
                flush=True,
            )
        else:
            batches = [
                {key: value.to(args.device) for key, value in batch.items()}
                for batch in batches
            ]

        # Forward calibration sees only these inputs, never the labels. This
        # slice is small and is replayed by every backend, so it stays resident.
        calibration_batches = batches[: max(1, args.calibration_batches)]
        args.calibration_inputs = [
            {"input_ids": b["input_ids"].to(args.device)}
            if is_gpt2
            else {"pixel_values": b["pixel_values"].to(args.device)}
            for b in calibration_batches
        ]
        args.calibration_sample_count = sum(
            int(b["input_ids"].numel() if is_gpt2 else b["pixel_values"].shape[0])
            for b in calibration_batches
        )

        # Streaming compares each batch against a second, unconverted model and
        # throws the logits away, instead of holding the whole split twice. It
        # is the only way the full WikiText-2 run fits in host memory.
        streaming = bool(args.stream_metrics) and is_gpt2
        reference_model: nn.Module | None = None
        if streaming:
            print(f"[{model_name}] streaming metrics: keeping a second exact model "
                  f"instead of {len(batches) * args.gpt2_sequence_length * 50257 * 4 / 1024**3:.0f} "
                  f"GiB of cached logits", flush=True)
            reference_model = builder(args)
            reference_model.eval()

        reference_logits: torch.Tensor | None = None
        for scope in args.scopes:
            for backend in args.backends:
                if backend == "exact" and scope != args.scopes[0]:
                    continue  # The exact model does not depend on the scope.
                print(f"[{model_name}] scope={scope} backend={backend} ...", flush=True)
                model = builder(args)
                coverage = convert(model, backend, scope, args, is_gpt2)

                if streaming:
                    outcome = run_gpt2_streaming(
                        model, None if backend == "exact" else reference_model, batches
                    )
                    elapsed = float(outcome.pop("approximate_forward_seconds"))
                    streamed = outcome.pop("streamed_metrics", None)
                    logits = None
                else:
                    started = time.perf_counter()
                    outcome = runner(model, batches)
                    elapsed = time.perf_counter() - started
                    streamed = None
                    logits = outcome.pop("logits").cpu()
                    if backend == "exact":
                        reference_logits = logits
                if args.save_logits and logits is not None:
                    # Caching the raw outputs means a change to the comparison
                    # metrics never costs another emulated forward pass, which
                    # is the expensive part by orders of magnitude. It is only
                    # worth it when the outputs are small: a language model's
                    # logits are tokens x vocabulary, which for GPT-2 at 25k
                    # tokens is 4.8 GB per condition and fills a disk long
                    # before it saves any time.
                    gigabytes = logits.numel() * 4 / 1024**3
                    if gigabytes > args.max_logit_cache_gb:
                        print(
                            f"    (skipping logit cache: {gigabytes:.2f} GB exceeds "
                            f"--max-logit-cache-gb={args.max_logit_cache_gb})",
                            flush=True,
                        )
                    else:
                        cache = args.output.parent / f"{args.output.stem}_{model_name}_{scope}_{backend}.pt"
                        torch.save(logits.to(torch.float32), cache)

                entry: dict[str, object] = {
                    "model": model_name,
                    "scope": "none" if backend == "exact" else scope,
                    "backend": backend,
                    "coverage": coverage,
                    "emulated_forward_seconds": elapsed,
                    **outcome,
                }
                if backend in COSTED_BACKENDS:
                    # An analytical proxy for the weighted product, recorded so
                    # the trade-off tables never have to recompute it from the
                    # configuration by hand. The nonlinear tables are costed
                    # separately and are not part of this record.
                    entry["multiplier_cost"] = multiplier_cost_summary(
                        backend,
                        args.prefix_bits,
                        args.dyadic_terms,
                        args.mantissa_bits,
                        args.max_shift,
                    )
                if streamed is not None:
                    entry.update(streamed)
                elif reference_logits is not None and backend != "exact" and logits is not None:
                    entry.update(compare_to_reference(logits, reference_logits))
                results.append(entry)
                args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
                summary = ", ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in entry.items()
                    if k in {"top1", "top5", "perplexity", "argmax_agreement", "logit_nrmse", "output_gain"}
                )
                print(f"    {summary}", flush=True)
                del model

    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
