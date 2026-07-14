"""
Torch-native B-PLA sensitivity probe for ViT.

This follows the old Mitchell C-2 ViT experiment structurally, but uses the
CUDA-friendly B-PLA proxy modules to check whether ViT accuracy is sensitive to
approximating Linear projections and GELU-like activations.
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
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from modules.torch_bpla import (
    SharedBPLATables,
    TorchBPLAActivation,
    TorchBPLAConfig,
    TorchBPLALinear,
    calibrate_model_activation_range,
    replace_attention_matmuls,
    replace_linear_and_gelu,
)


IMAGENETTE_TO_IMAGENET = {
    0: 0,
    1: 217,
    2: 482,
    3: 491,
    4: 497,
    5: 566,
    6: 569,
    7: 571,
    8: 574,
    9: 701,
}


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


def make_dry_run_model(device: torch.device) -> ViTForImageClassification:
    config = ViTConfig(
        image_size=32,
        patch_size=16,
        num_channels=3,
        num_labels=10,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        hidden_act="gelu",
    )
    return ViTForImageClassification(config).to(device).eval()


def replace_vit_intermediate_activations(
    module: nn.Module,
    config: TorchBPLAConfig,
    tables: SharedBPLATables,
) -> int:
    replaced = 0
    for child in module.modules():
        if isinstance(child, TorchBPLAActivation):
            continue
        if hasattr(child, "intermediate_act_fn") and not isinstance(child.intermediate_act_fn, TorchBPLAActivation):
            child.intermediate_act_fn = TorchBPLAActivation("gelu", config, tables)
            replaced += 1
    return replaced


def count_bpla_activations(module: nn.Module) -> int:
    return sum(1 for child in module.modules() if isinstance(child, TorchBPLAActivation))


def list_replaced_modules(module: nn.Module) -> tuple[list[str], list[str]]:
    linear_names = []
    activation_names = []
    for name, child in module.named_modules():
        if isinstance(child, TorchBPLALinear):
            linear_names.append(name)
        elif isinstance(child, TorchBPLAActivation):
            activation_names.append(name)
    return linear_names, activation_names


def convert_model(
    model: ViTForImageClassification,
    args: argparse.Namespace,
    cfg: TorchBPLAConfig | None = None,
) -> tuple[ViTForImageClassification, int, int, int]:
    cfg = cfg or build_config(args)
    tables = SharedBPLATables(cfg)
    replaced_linear = replace_linear_and_gelu(
        model,
        cfg,
        replace_linear=not args.no_linear,
        replace_gelu=not args.no_gelu,
        max_linear_modules=args.max_linear_modules,
        tables=tables,
    )
    replaced_act_fn = 0
    if not args.no_gelu:
        replaced_act_fn = replace_vit_intermediate_activations(model, cfg, tables)
    replaced_attention = 0
    if not args.no_attention:
        replaced_attention = replace_attention_matmuls(model, cfg, tables)
    return model, replaced_linear, max(replaced_act_fn, count_bpla_activations(model)), replaced_attention


def prepare_imagenette(args: argparse.Namespace):
    processor = ViTImageProcessor.from_pretrained(args.model_id)
    raw = load_dataset(args.dataset_id)
    if "validation" in raw:
        split = raw["validation"]
    elif "test" in raw:
        split = raw["test"]
    else:
        split = raw["train"].train_test_split(test_size=0.3, seed=42)["test"]

    def transform(batch):
        images = [image.convert("RGB") for image in batch["image"]]
        inputs = processor(images, return_tensors="pt")
        labels = [IMAGENETTE_TO_IMAGENET[int(label)] for label in batch["label"]]
        inputs["labels"] = torch.tensor(labels, dtype=torch.long)
        return inputs

    return split.with_transform(transform)


def evaluate(model: ViTForImageClassification, loader, device: torch.device, num_samples: int) -> tuple[float, float, torch.Tensor]:
    model.eval()
    total = 0
    top1 = 0
    top5 = 0
    logits_all = []
    batch_size = loader.batch_size if loader.batch_size is not None else 1
    total_batches = min(len(loader), math.ceil(num_samples / batch_size))
    with torch.no_grad():
        for batch in tqdm(loader, desc="ViT eval", total=total_batches):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            logits = model(pixel_values).logits
            logits_all.append(logits.detach().cpu())
            pred5 = logits.topk(5, dim=-1).indices
            correct = pred5.eq(labels.view(-1, 1).expand_as(pred5))
            top1 += correct[:, :1].sum().item()
            top5 += correct.sum().item()
            total += labels.numel()
            if total >= num_samples:
                break
    return 100.0 * top1 / total, 100.0 * top5 / total, torch.cat(logits_all, dim=0)


def compare_logits(model_a: nn.Module, model_b: nn.Module, pixel_values: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        logits_a = model_a(pixel_values).logits
        logits_b = model_b(pixel_values).logits
    diff = logits_b - logits_a
    return {
        "mae": diff.abs().mean().item(),
        "rmse": torch.sqrt((diff * diff).mean()).item(),
        "top1_agreement": (logits_a.argmax(dim=-1) == logits_b.argmax(dim=-1)).float().mean().item() * 100.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ViT B-PLA torch-native probe.")
    parser.add_argument("--model-id", default="google/vit-base-patch16-224")
    parser.add_argument("--dataset-id", default="johnowhitaker/imagenette2-320")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prefix-bits", type=int, default=4)
    parser.add_argument("--affine-path", choices=["float", "dyadic"], default="dyadic")
    parser.add_argument("--dyadic-terms", type=int, default=2)
    parser.add_argument("--max-shift", type=int, default=16)
    parser.add_argument("--activation-range", type=float, default=4.0)
    parser.add_argument("--calibration-batches", type=int, default=4)
    parser.add_argument("--no-calibrate-activation", action="store_true")
    parser.add_argument("--linear-chunk-out", type=int, default=32)
    parser.add_argument("--max-linear-modules", type=int, default=None)
    parser.add_argument("--no-linear", action="store_true")
    parser.add_argument("--no-gelu", action="store_true")
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--evaluate-ann", action="store_true")
    parser.add_argument("--list-replaced-modules", action="store_true")
    parser.add_argument("--stop-after-conversion", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    if args.dry_run:
        ann = make_dry_run_model(device)
        probe = copy.deepcopy(ann)
        probe, replaced_linear, replaced_act, replaced_attention = convert_model(probe, args)
        pixels = torch.randn(2, 3, 32, 32, device=device)
        stats = compare_logits(ann, probe, pixels)
        print("\nViT B-PLA dry run")
        print("=" * 72)
        print(f"replaced Linear modules  : {replaced_linear}")
        print(f"replaced act callables   : {replaced_act}")
        print(f"replaced attention blocks: {replaced_attention}")
        if args.list_replaced_modules:
            linear_names, activation_names = list_replaced_modules(probe)
            print("replaced Linear names    :")
            for name in linear_names:
                print(f"  - {name}")
            print("replaced activation names:")
            for name in activation_names:
                print(f"  - {name}")
        print(f"logit MAE / RMSE         : {stats['mae']:.6e} / {stats['rmse']:.6e}")
        print(f"top-1 agreement          : {stats['top1_agreement']:.2f}%")
        return

    print(f"Loading ViT model: {args.model_id}")
    ann = ViTForImageClassification.from_pretrained(args.model_id).to(device).eval()
    dataset = prepare_imagenette(args)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    cfg = build_config(args)
    if not args.no_gelu and not args.no_calibrate_activation:
        measured_range = calibrate_model_activation_range(
            ann,
            loader,
            lambda model, batch: model(batch["pixel_values"].to(device)),
            args.calibration_batches,
        )
        cfg = TorchBPLAConfig(**{**cfg.__dict__, "activation_range": measured_range})
        print(f"Calibrated shared GELU range: +/-{measured_range:.6f}")
    probe = copy.deepcopy(ann)
    probe, replaced_linear, replaced_act, replaced_attention = convert_model(probe, args, cfg)
    print(f"Replaced Linear modules: {replaced_linear}")
    print(f"Replaced activation callables: {replaced_act}")
    print(f"Replaced attention blocks: {replaced_attention}")
    if args.list_replaced_modules:
        linear_names, activation_names = list_replaced_modules(probe)
        print("Replaced Linear module names:")
        for name in linear_names:
            print(f"  - {name}")
        print("Replaced activation module names:")
        for name in activation_names:
            print(f"  - {name}")
    if replaced_linear > 16:
        print(
            "Warning: many B-PLA Linear modules are enabled. This proxy expands "
            "matmul into elementwise approximate products and can be very slow. "
            "Use --no-linear or --max-linear-modules for quick sensitivity probes."
        )

    if args.stop_after_conversion:
        image_size = ann.config.image_size if isinstance(ann.config.image_size, int) else ann.config.image_size[0]
        pixels = torch.randn(1, 3, image_size, image_size, device=device)
        stats = compare_logits(ann, probe, pixels)
        print("Stop-after-conversion smoke forward passed.")
        print(f"logit MAE / RMSE: {stats['mae']:.6e} / {stats['rmse']:.6e}")
        return

    ann_logits = None
    if args.evaluate_ann:
        ann_top1, ann_top5, ann_logits = evaluate(ann, loader, device, args.num_samples)
        print(f"ANN Top-1 / Top-5: {ann_top1:.2f}% / {ann_top5:.2f}%")

    top1, top5, bpla_logits = evaluate(probe, loader, device, args.num_samples)
    print("\nViT B-PLA Probe")
    print("=" * 72)
    print(f"affine path              : {args.affine_path} (terms={args.dyadic_terms}, prefix={args.prefix_bits})")
    print(f"replaced Linear modules  : {replaced_linear}")
    print(f"replaced act callables   : {replaced_act}")
    print(f"replaced attention blocks: {replaced_attention}")
    print(f"B-PLA Top-1 / Top-5      : {top1:.2f}% / {top5:.2f}%")
    if ann_logits is not None:
        if ann_logits.shape != bpla_logits.shape:
            raise RuntimeError(
                f"ANN/B-PLA logit shapes differ: {tuple(ann_logits.shape)} vs {tuple(bpla_logits.shape)}"
            )
        diff = bpla_logits - ann_logits
        agreement = (ann_logits.argmax(dim=-1) == bpla_logits.argmax(dim=-1)).float().mean().item() * 100.0
        mae = diff.abs().mean().item()
        rmse = torch.sqrt((diff * diff).mean()).item()
        print(f"ANN-BPLA Top-1 agreement : {agreement:.2f}%")
        print(f"ANN-BPLA logit MAE/RMSE  : {mae:.6e} / {rmse:.6e}")


if __name__ == "__main__":
    main()
