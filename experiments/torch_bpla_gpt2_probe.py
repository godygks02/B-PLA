"""
Torch-native B-PLA sensitivity probe for GPT-2.

This script mirrors the old Mitchell C-2 GPT-2 experiment at the level needed
for a fast accuracy-sensitivity check: GPT-2 Conv1D projections and GELU are
replaced with CUDA-friendly B-PLA proxy modules. It deliberately does not claim
hardware-faithful RTL behavior.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from modules.torch_bpla import (
    SharedBPLATables,
    TorchBPLAActivation,
    TorchBPLAConfig,
    calibrate_model_activation_range,
    replace_gpt2_conv1d_and_gelu,
)


DATASET_ALIASES = {
    # Newer huggingface_hub releases require a namespace/name repository ID.
    "wikitext": "Salesforce/wikitext",
}


def normalize_dataset_name(dataset_name: str) -> str:
    return DATASET_ALIASES.get(dataset_name, dataset_name)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_config(args: argparse.Namespace) -> TorchBPLAConfig:
    return TorchBPLAConfig(
        prefix_bits=args.prefix_bits,
        affine_path=args.affine_path,
        dyadic_terms=args.dyadic_terms,
        max_shift=args.max_shift,
        activation_range=args.activation_range,
        linear_chunk_out=args.linear_chunk_out,
    )


def make_dry_run_model(device: torch.device) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=128,
        n_positions=32,
        n_ctx=32,
        n_embd=32,
        n_layer=2,
        n_head=4,
        bos_token_id=0,
        eos_token_id=1,
    )
    return GPT2LMHeadModel(config).to(device).eval()


def convert_model(
    model: GPT2LMHeadModel,
    args: argparse.Namespace,
    cfg: TorchBPLAConfig | None = None,
) -> tuple[GPT2LMHeadModel, int, int]:
    cfg = cfg or build_config(args)
    tables = SharedBPLATables(cfg)
    replaced = replace_gpt2_conv1d_and_gelu(
        model,
        cfg,
        replace_conv1d=not args.no_conv1d,
        replace_gelu=not args.no_gelu,
        max_conv1d_modules=args.max_conv1d_modules,
        tables=tables,
    )
    replaced_activations = sum(1 for child in model.modules() if isinstance(child, TorchBPLAActivation))
    return model, replaced, replaced_activations


def evaluate_ppl(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    dataset,
    device: torch.device,
    max_length: int,
    stride: int,
    num_windows: int,
) -> float:
    model.eval()
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    total_nll = torch.zeros((), device=device)
    total_tokens = 0
    prev_end_loc = 0

    for window_idx, begin_loc in enumerate(tqdm(range(0, seq_len, stride), desc="GPT-2 PPL")):
        if window_idx >= num_windows:
            break
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        valid_tokens = int((target_ids[:, 1:] != -100).sum().item())
        total_nll += loss.detach() * valid_tokens
        total_tokens += valid_tokens
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    if total_tokens == 0:
        raise RuntimeError("No target tokens were evaluated for perplexity.")
    return torch.exp(total_nll / total_tokens).item()


def evaluate_ann_bpla_pair(
    ann: GPT2LMHeadModel,
    bpla: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    dataset,
    device: torch.device,
    max_length: int,
    stride: int,
    num_windows: int,
) -> dict[str, float]:
    """Evaluate ANN and B-PLA on identical windows and compare their logits."""

    ann.eval()
    bpla.eval()
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    ann_total_nll = torch.zeros((), device=device)
    bpla_total_nll = torch.zeros((), device=device)
    total_tokens = 0
    agreement_count = 0
    logit_abs_sum = 0.0
    logit_sq_sum = 0.0
    logit_count = 0
    prev_end_loc = 0

    for window_idx, begin_loc in enumerate(tqdm(range(0, seq_len, stride), desc="GPT-2 ANN/B-PLA")):
        if window_idx >= num_windows:
            break
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            ann_outputs = ann(input_ids, labels=target_ids)
            bpla_outputs = bpla(input_ids, labels=target_ids)

        valid_mask = target_ids[:, 1:] != -100
        valid_tokens = int(valid_mask.sum().item())
        ann_total_nll += ann_outputs.loss.detach() * valid_tokens
        bpla_total_nll += bpla_outputs.loss.detach() * valid_tokens
        total_tokens += valid_tokens

        ann_logits = ann_outputs.logits[:, :-1, :][valid_mask]
        bpla_logits = bpla_outputs.logits[:, :-1, :][valid_mask]
        agreement_count += int((ann_logits.argmax(dim=-1) == bpla_logits.argmax(dim=-1)).sum().item())
        diff = bpla_logits - ann_logits
        logit_abs_sum += diff.abs().sum().item()
        logit_sq_sum += (diff * diff).sum().item()
        logit_count += diff.numel()

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    if total_tokens == 0 or logit_count == 0:
        raise RuntimeError("No target tokens were evaluated for ANN/B-PLA comparison.")

    ann_ppl = torch.exp(ann_total_nll / total_tokens).item()
    bpla_ppl = torch.exp(bpla_total_nll / total_tokens).item()
    return {
        "ann_ppl": ann_ppl,
        "bpla_ppl": bpla_ppl,
        "ppl_delta": bpla_ppl - ann_ppl,
        "ppl_delta_pct": 100.0 * (bpla_ppl - ann_ppl) / ann_ppl,
        "next_token_agreement": 100.0 * agreement_count / total_tokens,
        "logit_mae": logit_abs_sum / logit_count,
        "logit_rmse": math.sqrt(logit_sq_sum / logit_count),
        "evaluated_tokens": float(total_tokens),
    }


def compare_logits(model_a: nn.Module, model_b: nn.Module, input_ids: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        logits_a = model_a(input_ids).logits
        logits_b = model_b(input_ids).logits
    diff = logits_b - logits_a
    return {
        "mae": diff.abs().mean().item(),
        "rmse": torch.sqrt((diff * diff).mean()).item(),
        "next_token_agreement": (logits_a.argmax(dim=-1) == logits_b.argmax(dim=-1)).float().mean().item() * 100.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT-2 B-PLA torch-native probe.")
    parser.add_argument("--model-id", default="gpt2")
    parser.add_argument("--dataset-name", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--num-windows", type=int, default=4)
    parser.add_argument("--prefix-bits", type=int, default=4)
    parser.add_argument("--affine-path", choices=["float", "dyadic"], default="dyadic")
    parser.add_argument("--dyadic-terms", type=int, default=2)
    parser.add_argument("--max-shift", type=int, default=16)
    parser.add_argument("--activation-range", type=float, default=4.0)
    parser.add_argument("--calibration-batches", type=int, default=4)
    parser.add_argument("--no-calibrate-activation", action="store_true")
    parser.add_argument("--linear-chunk-out", type=int, default=32)
    parser.add_argument("--max-conv1d-modules", type=int, default=None)
    parser.add_argument("--no-conv1d", action="store_true")
    parser.add_argument("--no-gelu", action="store_true")
    parser.add_argument("--evaluate-ann", action="store_true")
    parser.add_argument("--stop-after-conversion", action="store_true")
    parser.add_argument("--smoke-text", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    if args.dry_run:
        ann = make_dry_run_model(device)
        probe = copy.deepcopy(ann)
        probe, replaced, replaced_activations = convert_model(probe, args)
        input_ids = torch.randint(0, ann.config.vocab_size, (1, 16), device=device)
        stats = compare_logits(ann, probe, input_ids)
        print("\nGPT-2 B-PLA dry run")
        print("=" * 72)
        print(f"replaced Conv1D modules  : {replaced}")
        print(f"replaced act callables   : {replaced_activations}")
        print(f"logit MAE / RMSE         : {stats['mae']:.6e} / {stats['rmse']:.6e}")
        print(f"next-token agreement     : {stats['next_token_agreement']:.2f}%")
        return

    print(f"Loading model/tokenizer: {args.model_id}")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_id)
    ann = GPT2LMHeadModel.from_pretrained(args.model_id).to(device).eval()
    dataset_name = normalize_dataset_name(args.dataset_name)
    if dataset_name != args.dataset_name:
        print(f"Resolved dataset alias: {args.dataset_name} -> {dataset_name}")
    print(f"Loading dataset: {dataset_name}/{args.dataset_config}/{args.dataset_split}")
    dataset = load_dataset(dataset_name, args.dataset_config, split=args.dataset_split)
    cfg = build_config(args)
    if not args.no_gelu and not args.no_calibrate_activation:
        texts = (text for text in dataset["text"] if text.strip())
        calibration_inputs = (
            tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length).input_ids
            for text in texts
        )
        measured_range = calibrate_model_activation_range(
            ann,
            calibration_inputs,
            lambda model, input_ids: model(input_ids.to(device)),
            args.calibration_batches,
        )
        cfg = TorchBPLAConfig(**{**cfg.__dict__, "activation_range": measured_range})
        print(f"Calibrated shared GELU range: +/-{measured_range:.6f}")
    probe = copy.deepcopy(ann)
    probe, replaced, replaced_activations = convert_model(probe, args, cfg)
    print(f"Replaced Conv1D modules: {replaced}")
    print(f"Replaced activation callables: {replaced_activations}")

    if args.stop_after_conversion:
        sample = tokenizer(args.smoke_text, return_tensors="pt").input_ids[:, : min(32, args.max_length)].to(device)
        stats = compare_logits(ann, probe, sample)
        print("Stop-after-conversion smoke forward passed.")
        print(f"smoke text: {args.smoke_text!r}")
        print(f"logit MAE / RMSE: {stats['mae']:.6e} / {stats['rmse']:.6e}")
        return

    comparison = None
    if args.evaluate_ann:
        comparison = evaluate_ann_bpla_pair(
            ann,
            probe,
            tokenizer,
            dataset,
            device,
            args.max_length,
            args.stride,
            args.num_windows,
        )
        bpla_ppl = comparison["bpla_ppl"]
    else:
        bpla_ppl = evaluate_ppl(probe, tokenizer, dataset, device, args.max_length, args.stride, args.num_windows)
    print("\nGPT-2 B-PLA Probe")
    print("=" * 72)
    print(f"affine path              : {args.affine_path} (terms={args.dyadic_terms}, prefix={args.prefix_bits})")
    print(f"replaced Conv1D modules  : {replaced}")
    print(f"replaced act callables   : {replaced_activations}")
    if comparison is not None:
        print(f"ANN PPL                  : {comparison['ann_ppl']:.4f}")
    print(f"B-PLA proxy PPL          : {bpla_ppl:.4f}")
    if comparison is not None:
        print(f"B-PLA PPL delta          : {comparison['ppl_delta']:+.4f} ({comparison['ppl_delta_pct']:+.2f}%)")
        print(f"ANN-BPLA token agreement : {comparison['next_token_agreement']:.2f}%")
        print(f"ANN-BPLA logit MAE/RMSE  : {comparison['logit_mae']:.6e} / {comparison['logit_rmse']:.6e}")
        print(f"evaluated target tokens  : {int(comparison['evaluated_tokens'])}")


if __name__ == "__main__":
    main()
