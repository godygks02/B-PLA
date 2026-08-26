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
BACKENDS = ("exact", "ptq-w8a8", "ptq-w8a8-static", "pao", "bpla-float", "bpla-dyadic")

#: The two W8A8 recipes, kept as separate backends so a single matched run can
#: report both against the same reference. ``ptq-w8a8`` uses dynamic per-token
#: activation scales, which is what ZeroQuant, LLM.int8() and SmoothQuant's O1
#: setting all do and is the strong form of the baseline. ``ptq-w8a8-static``
#: uses static per-tensor scales from percentile calibration, the conventional
#: recipe. Reporting only the first would look like a straw man in reverse;
#: reporting only the second would be the straw man.
PTQ_GRANULARITY = {"ptq-w8a8": "token", "ptq-w8a8-static": "tensor"}

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
        activation_range=args.activation_range,
        linear_chunk_out=args.linear_chunk_out,
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
        if scope not in PTQ_SCOPES:
            raise ValueError(
                f"The {backend} backend has no {scope!r} scope: W8A8 leaves GELU, "
                "Softmax and LayerNorm in floating point by construction. Run it "
                "at the multiplication scope and compare there."
            )
        config = TorchPTQConfig(
            weight_bits=args.ptq_weight_bits,
            activation_bits=args.ptq_activation_bits,
            per_channel_weights=args.ptq_per_channel_weights,
            activation_percentile=args.ptq_percentile,
            activation_granularity=PTQ_GRANULARITY[backend],
        )
        if is_gpt2:
            record["linear_modules"] = replace_ptq_gpt2_conv1d(
                model, config, replace_lm_head=args.replace_lm_head
            )
        else:
            record["linear_modules"] = replace_ptq_linear(
                model, config, replace_conv2d=args.replace_conv2d
            )
        record["attention_blocks"] = replace_ptq_attention_matmuls(model, config, mode="ptq-full")
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

    if backend == "pao":
        config = TorchPAOConfig(matmul_chunk_out=args.linear_chunk_out, alpha=args.pao_alpha)
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

    config = _bpla_config(args, "float" if backend == "bpla-float" else "dyadic")
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
    from torch_bpla_vit_probe import IMAGENETTE_TO_IMAGENET

    processor = ViTImageProcessor.from_pretrained(args.vit_model_id)
    raw = load_dataset(args.vit_dataset_id)
    if "validation" in raw:
        split = raw["validation"]
    elif "test" in raw:
        split = raw["test"]
    else:
        split = raw["train"].train_test_split(test_size=0.3, seed=42)["test"]

    def transform(batch):
        images = [image.convert("RGB") for image in batch["image"]]
        inputs = processor(images, return_tensors="pt")
        inputs["labels"] = torch.tensor(
            [IMAGENETTE_TO_IMAGENET[int(label)] for label in batch["label"]], dtype=torch.long
        )
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


@torch.no_grad()
def run_vit(model: nn.Module, batches: list[dict[str, torch.Tensor]]) -> dict[str, object]:
    logits = []
    labels = []
    for batch in batches:
        logits.append(model(batch["pixel_values"]).logits.float().cpu())
        labels.append(batch["labels"].cpu())
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
    for batch in batches:
        input_ids = batch["input_ids"]
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


def compare_to_reference(current: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Compare approximate outputs to the exact ones.

    Logit metrics are computed on *row-centered* logits. Softmax is invariant to
    a per-row constant, and for GPT-2 that constant carries 99.95% of the raw
    logit energy, so an uncentered gain or MAE mostly measures a shift the model
    never sees. The uncentered figures are still reported, clearly labelled, for
    comparison with work that quotes them.
    """

    # Accumulated over row chunks rather than over whole float64 copies. Every
    # metric here is a ratio of sums, so chunking changes nothing but the peak
    # memory -- and that peak is what decides whether the run finishes: a
    # language model's logits are tokens x vocabulary, so at 12,800 GPT-2 tokens
    # one float64 copy is 5.1 GB and the original four-copy form needed roughly
    # 20 GB to compare a run that had already completed its forward passes.
    rows = current.shape[0]
    chunk = max(1, min(rows, _COMPARISON_CHUNK_ELEMENTS // max(1, current.shape[-1])))

    totals = dict.fromkeys(
        (
            "elements",
            "abs_difference",
            "squared_difference",
            "squared_reference",
            "cross",
            "uncentered_abs_difference",
            "uncentered_cross",
            "uncentered_squared_reference",
        ),
        0.0,
    )
    agreements = 0

    for start in range(0, rows, chunk):
        current_chunk = current[start : start + chunk].double()
        reference_chunk = reference[start : start + chunk].double()
        centered_current = current_chunk - current_chunk.mean(dim=-1, keepdim=True)
        centered_reference = reference_chunk - reference_chunk.mean(dim=-1, keepdim=True)
        centered_difference = centered_current - centered_reference

        totals["elements"] += float(current_chunk.numel())
        totals["abs_difference"] += float(centered_difference.abs().sum())
        totals["squared_difference"] += float(centered_difference.pow(2).sum())
        totals["squared_reference"] += float(centered_reference.pow(2).sum())
        totals["cross"] += float((centered_current * centered_reference).sum())
        totals["uncentered_abs_difference"] += float((current_chunk - reference_chunk).abs().sum())
        totals["uncentered_cross"] += float((current_chunk * reference_chunk).sum())
        totals["uncentered_squared_reference"] += float(reference_chunk.pow(2).sum())
        agreements += int(
            (current_chunk.argmax(dim=-1) == reference_chunk.argmax(dim=-1)).sum()
        )

    elements = totals["elements"]
    mean_squared_difference = totals["squared_difference"] / elements
    mean_squared_reference = totals["squared_reference"] / elements

    return {
        "logit_mae": totals["abs_difference"] / elements,
        "logit_rmse": math.sqrt(mean_squared_difference),
        "logit_nrmse": math.sqrt(mean_squared_difference) / math.sqrt(mean_squared_reference),
        "argmax_agreement": 100.0 * agreements / rows,
        # The model-level counterpart of the per-product gain: a systematic
        # contraction shows up here even when the argmax survives.
        "output_gain": totals["cross"] / totals["squared_reference"],
        "uncentered_logit_mae": totals["uncentered_abs_difference"] / elements,
        "uncentered_output_gain": (
            totals["uncentered_cross"] / totals["uncentered_squared_reference"]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact / PAO / B-PLA training-free comparison.")
    parser.add_argument("--models", nargs="+", default=["vit"], choices=["vit", "gpt2"])
    parser.add_argument("--backends", nargs="+", default=list(BACKENDS), choices=BACKENDS)
    parser.add_argument("--scopes", nargs="+", default=["multiplication"], choices=SCOPES)

    parser.add_argument("--vit-model-id", default="google/vit-base-patch16-224")
    parser.add_argument("--vit-dataset-id", default="johnowhitaker/imagenette2-320")
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
    parser.add_argument("--activation-range", type=float, default=4.0)
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
        default=None,
        help="Optional PAO error-compensation constant (Sec. 2.7). Off by default, as in the paper.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for both the model and the evaluation batches.",
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
        ],
        "results": [],
    }
    results: list[dict[str, object]] = record["results"]  # type: ignore[assignment]

    # Fail before the first checkpoint is downloaded rather than hours into a
    # run: an unsupported combination costs the whole queue if it surfaces late.
    if any(backend in PTQ_GRANULARITY for backend in args.backends):
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
        batches = [
            {key: value.to(args.device) for key, value in batch.items()}
            for batch in batches
        ]

        # Forward calibration sees only these inputs, never the labels.
        calibration_batches = batches[: max(1, args.calibration_batches)]
        args.calibration_inputs = [
            {"input_ids": b["input_ids"]} if is_gpt2 else {"pixel_values": b["pixel_values"]}
            for b in calibration_batches
        ]
        args.calibration_sample_count = sum(
            int(b["input_ids"].numel() if is_gpt2 else b["pixel_values"].shape[0])
            for b in calibration_batches
        )

        reference_logits: torch.Tensor | None = None
        for scope in args.scopes:
            for backend in args.backends:
                if backend == "exact" and scope != args.scopes[0]:
                    continue  # The exact model does not depend on the scope.
                print(f"[{model_name}] scope={scope} backend={backend} ...", flush=True)
                model = builder(args)
                coverage = convert(model, backend, scope, args, is_gpt2)
                started = time.perf_counter()
                outcome = runner(model, batches)
                elapsed = time.perf_counter() - started
                logits = outcome.pop("logits").cpu()
                if backend == "exact":
                    reference_logits = logits
                if args.save_logits:
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
                if reference_logits is not None and backend != "exact":
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
