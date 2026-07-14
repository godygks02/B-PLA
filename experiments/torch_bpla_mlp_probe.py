"""
Fast B-PLA proxy probe on the pretrained MNIST MLP.

Unlike `bpla_mlp_experiment.py`, this script stays entirely in PyTorch. It is
intended as the template for GPT-2/ViT sensitivity checks where CUDA support and
reasonable runtime matter more than hardware-faithful integer simulation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PROJECT_ROOT.parent / "Universal_Training_free_ANN2SNN_conversion"
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(REFERENCE_ROOT))

from model_utils import ToyTransformerMLP, get_device
from modules.torch_bpla import TorchBPLAConfig, replace_linear_and_gelu
from modules.compute_energy import (
    BPLAComputeConfig,
    ComputeEnergyTablePJ,
    estimate_workload_compute_energy,
    format_compute_energy_report,
    mlp_workload,
)


def load_mnist(batch_size: int, max_test_samples: int | None) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )
    dataset = datasets.MNIST(root=str(REFERENCE_ROOT / "data"), train=False, download=False, transform=transform)
    if max_test_samples is not None:
        dataset = Subset(dataset, range(min(max_test_samples, len(dataset))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, torch.Tensor]:
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    logits_all = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            logits_all.append(logits.detach().cpu())
            correct += (logits.argmax(dim=-1) == batch_y).sum().item()
            total += batch_y.numel()
    return 100.0 * correct / total, torch.cat(logits_all, dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast torch-native B-PLA sensitivity probe.")
    parser.add_argument("--checkpoint", default=str(REFERENCE_ROOT / "test_model" / "mnist_mlp.pth"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-test-samples", type=int, default=1000)
    parser.add_argument("--prefix-bits", type=int, default=4)
    parser.add_argument("--affine-path", choices=["float", "dyadic"], default="dyadic")
    parser.add_argument("--dyadic-terms", type=int, default=2)
    parser.add_argument("--linear-chunk-out", type=int, default=32)
    parser.add_argument("--activation-range", type=float, default=4.0)
    parser.add_argument("--no-linear", action="store_true")
    parser.add_argument("--no-gelu", action="store_true")
    parser.add_argument("--max-linear-modules", type=int, default=None)
    parser.add_argument("--energy-shift-pj", type=float, default=0.0)
    parser.add_argument("--energy-control-pj", type=float, default=0.005)
    parser.add_argument("--energy-tanh-pj", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    loader = load_mnist(args.batch_size, args.max_test_samples)

    ann = ToyTransformerMLP(checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["num_classes"])
    ann.load_state_dict(checkpoint["state_dict"])
    ann.eval()

    probe = ToyTransformerMLP(checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["num_classes"])
    probe.load_state_dict(checkpoint["state_dict"])
    cfg = TorchBPLAConfig(
        prefix_bits=args.prefix_bits,
        affine_path=args.affine_path,
        dyadic_terms=args.dyadic_terms,
        activation_range=args.activation_range,
        linear_chunk_out=args.linear_chunk_out,
    )
    replaced = replace_linear_and_gelu(
        probe,
        cfg,
        replace_linear=not args.no_linear,
        replace_gelu=not args.no_gelu,
        max_linear_modules=args.max_linear_modules,
    )

    ann_acc, ann_logits = evaluate(ann, loader, device)
    probe_acc, probe_logits = evaluate(probe, loader, device)
    diff = probe_logits - ann_logits
    mae = diff.abs().mean().item()
    rmse = torch.sqrt((diff * diff).mean()).item()
    agreement = (probe_logits.argmax(dim=-1) == ann_logits.argmax(dim=-1)).float().mean().item() * 100.0

    print("\nTorch B-PLA MLP Probe")
    print("=" * 72)
    print(f"device                   : {device}")
    print(f"samples                  : {ann_logits.shape[0]}")
    print(f"replaced linear modules  : {replaced}")
    print(f"replace GELU             : {not args.no_gelu}")
    print(f"affine path              : {args.affine_path} (terms={args.dyadic_terms}, prefix={args.prefix_bits})")
    print(f"ANN accuracy             : {ann_acc:.2f}%")
    print(f"B-PLA proxy accuracy     : {probe_acc:.2f}%")
    print(f"accuracy drop            : {ann_acc - probe_acc:.2f}%")
    print(f"logit MAE / RMSE         : {mae:.6e} / {rmse:.6e}")
    print(f"prediction agreement     : {agreement:.2f}%")
    workload = mlp_workload(
        checkpoint["input_dim"],
        checkpoint["hidden_dim"],
        checkpoint["num_classes"],
        replace_linear=not args.no_linear,
        replace_gelu=not args.no_gelu,
        max_linear_modules=args.max_linear_modules,
    )
    energy = estimate_workload_compute_energy(
        workload,
        BPLAComputeConfig(args.affine_path, args.dyadic_terms),
        ComputeEnergyTablePJ(
            fixed_shift=args.energy_shift_pj,
            small_control=args.energy_control_pj,
            fp32_tanh=args.energy_tanh_pj,
        ),
    )
    print("\n" + format_compute_energy_report(energy))


if __name__ == "__main__":
    main()
