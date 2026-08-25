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


SCOPES = ("multiplication", "nonlinear", "combined")
BACKENDS = ("exact", "pao", "bpla-float", "bpla-dyadic")


def _bpla_config(args: argparse.Namespace, affine_path: str) -> TorchBPLAConfig:
    return TorchBPLAConfig(
        prefix_bits=args.prefix_bits,
        affine_path=affine_path,
        dyadic_terms=args.dyadic_terms,
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

    current = current.double()
    reference = reference.double()
    centered_current = current - current.mean(dim=-1, keepdim=True)
    centered_reference = reference - reference.mean(dim=-1, keepdim=True)
    centered_difference = centered_current - centered_reference

    return {
        "logit_mae": float(centered_difference.abs().mean()),
        "logit_rmse": float(centered_difference.pow(2).mean().sqrt()),
        "logit_nrmse": float(
            centered_difference.pow(2).mean().sqrt() / centered_reference.pow(2).mean().sqrt()
        ),
        "argmax_agreement": float(
            (current.argmax(dim=-1) == reference.argmax(dim=-1)).float().mean() * 100.0
        ),
        # The model-level counterpart of the per-product gain: a systematic
        # contraction shows up here even when the argmax survives.
        "output_gain": float(
            (centered_current * centered_reference).sum() / centered_reference.pow(2).sum()
        ),
        "uncentered_logit_mae": float((current - reference).abs().mean()),
        "uncentered_output_gain": float((current * reference).sum() / reference.pow(2).sum()),
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
            "Wall-clock is not reported: neither backend has native hardware support.",
            "Logit metrics are row-centered: softmax ignores a per-row constant, and for "
            "GPT-2 that constant holds 99.95% of the raw logit energy.",
        ],
        "results": [],
    }
    results: list[dict[str, object]] = record["results"]  # type: ignore[assignment]

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
