"""
MNIST MLP experiment for the current B-PLA software primitives.

This script is intentionally a software-faithful bridge rather than a fast
PyTorch implementation. It wraps the existing NumPy B-PLA multiplier and
activation modules to test whether the current arithmetic primitives can be
inserted into a pretrained MLP without retraining.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PROJECT_ROOT.parent / "Universal_Training_free_ANN2SNN_conversion"

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(REFERENCE_ROOT))

from modules.bpla_activation import bpla_activation, build_activation_table, build_dyadic_activation_table
from modules.bpla_multiplier import bpla_multiply, build_bpla_dyadic_coefficients
from model_utils import ToyTransformerMLP, get_device
from modules.compute_energy import (
    BPLAComputeConfig,
    ComputeEnergyTablePJ,
    estimate_workload_compute_energy,
    format_compute_energy_report,
    mlp_workload,
)


@dataclass(frozen=True)
class EvalResult:
    accuracy: float
    samples: int


class BPLALinear(nn.Module):
    """Linear layer using elementwise B-PLA multiplication before reduction."""

    def __init__(
        self,
        source: nn.Linear,
        prefix_bits: int = 4,
        affine_path: str = "float",
        dyadic_terms: int = 1,
        max_shift: int = 16,
    ):
        super().__init__()
        self.prefix_bits = prefix_bits
        self.affine_path = affine_path
        self.dyadic_terms = dyadic_terms
        self.max_shift = max_shift
        self.dyadic_coeffs = None
        if affine_path == "dyadic":
            self.dyadic_coeffs = build_bpla_dyadic_coefficients(
                prefix_bits=prefix_bits,
                terms_per_linear_coeff=dyadic_terms,
                offset_terms=dyadic_terms,
                max_shift=max_shift,
            )
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.register_buffer("weight", source.weight.detach().clone())
        if source.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", source.bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        x_np = x.detach().cpu().numpy().astype(np.float32)
        w_np = self.weight.detach().cpu().numpy().astype(np.float32)

        products = bpla_multiply(
            x_np[:, None, :],
            w_np[None, :, :],
            prefix_bits=self.prefix_bits,
            affine_path=self.affine_path,
            dyadic_coeffs=self.dyadic_coeffs,
            dyadic_terms=self.dyadic_terms,
            max_shift=self.max_shift,
        )
        out_np = products.sum(axis=-1)
        if self.bias is not None:
            out_np += self.bias.detach().cpu().numpy().astype(np.float64)
        return torch.from_numpy(out_np.astype(np.float32)).to(device)


class BPLAGELU(nn.Module):
    """GELU wrapper using the current FP32 bit-prefix B-PLA activation table."""

    def __init__(
        self,
        prefix_bits: int = 4,
        x_min: float = -4.0,
        x_max: float = 4.0,
        affine_path: str = "float",
        dyadic_terms: int = 1,
        max_shift: int = 16,
    ):
        super().__init__()
        self.affine_path = affine_path
        self.table = None
        self.dyadic_table = None
        if affine_path == "float":
            self.table = build_activation_table(
                "gelu",
                prefix_bits=prefix_bits,
                x_min=x_min,
                x_max=x_max,
            )
        elif affine_path == "dyadic":
            self.dyadic_table = build_dyadic_activation_table(
                "gelu",
                prefix_bits=prefix_bits,
                x_min=x_min,
                x_max=x_max,
                slope_terms=dyadic_terms,
                intercept_terms=dyadic_terms,
                max_shift=max_shift,
            )
        else:
            raise ValueError("affine_path must be 'float' or 'dyadic'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        out = bpla_activation(
            x.detach().cpu().numpy().astype(np.float32),
            affine_path=self.affine_path,
            table=self.table,
            dyadic_table=self.dyadic_table,
        )
        return torch.from_numpy(out.astype(np.float32)).to(device)


class BPLAMLP(nn.Module):
    def __init__(
        self,
        ann_model: ToyTransformerMLP,
        linear_prefix_bits: int = 4,
        activation_prefix_bits: int = 4,
        gelu_range: float = 4.0,
        affine_path: str = "float",
        dyadic_terms: int = 1,
        max_shift: int = 16,
        approx_linear: bool = True,
        approx_gelu: bool = True,
    ):
        super().__init__()
        self.fc1 = self._copy_linear(ann_model.fc1, linear_prefix_bits, approx_linear, affine_path, dyadic_terms, max_shift)
        self.ln1 = self._copy_layer_norm(ann_model.ln1)
        self.gelu1 = (
            BPLAGELU(activation_prefix_bits, -gelu_range, gelu_range, affine_path, dyadic_terms, max_shift)
            if approx_gelu
            else nn.GELU()
        )

        self.fc2 = self._copy_linear(ann_model.fc2, linear_prefix_bits, approx_linear, affine_path, dyadic_terms, max_shift)
        self.ln2 = self._copy_layer_norm(ann_model.ln2)
        self.gelu2 = (
            BPLAGELU(activation_prefix_bits, -gelu_range, gelu_range, affine_path, dyadic_terms, max_shift)
            if approx_gelu
            else nn.GELU()
        )

        self.fc3 = self._copy_linear(ann_model.fc3, linear_prefix_bits, approx_linear, affine_path, dyadic_terms, max_shift)

    @staticmethod
    def _copy_linear(
        source: nn.Linear,
        prefix_bits: int,
        approximate: bool,
        affine_path: str,
        dyadic_terms: int,
        max_shift: int,
    ) -> nn.Module:
        if approximate:
            return BPLALinear(
                source,
                prefix_bits=prefix_bits,
                affine_path=affine_path,
                dyadic_terms=dyadic_terms,
                max_shift=max_shift,
            )
        copied = nn.Linear(source.in_features, source.out_features, bias=source.bias is not None)
        with torch.no_grad():
            copied.weight.copy_(source.weight)
            if source.bias is not None:
                copied.bias.copy_(source.bias)
        return copied

    @staticmethod
    def _copy_layer_norm(source: nn.LayerNorm) -> nn.LayerNorm:
        copied = nn.LayerNorm(source.normalized_shape, eps=source.eps, elementwise_affine=source.elementwise_affine)
        with torch.no_grad():
            if source.elementwise_affine:
                copied.weight.copy_(source.weight)
                copied.bias.copy_(source.bias)
        return copied

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.gelu1(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.gelu2(x)
        return self.fc3(x)


def load_mnist(batch_size: int, max_test_samples: int | None) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )
    data_root = REFERENCE_ROOT / "data"
    test_dataset = datasets.MNIST(root=str(data_root), train=False, download=False, transform=transform)
    if max_test_samples is not None:
        test_dataset = Subset(test_dataset, range(min(max_test_samples, len(test_dataset))))
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> EvalResult:
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            predicted = logits.argmax(dim=1)
            total += batch_y.numel()
            correct += (predicted == batch_y).sum().item()
    return EvalResult(accuracy=100.0 * correct / total, samples=total)


def estimate_energy(
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    approx_linear: bool,
    approx_gelu: bool,
    affine_path: str,
    dyadic_terms: int,
    energy_table: ComputeEnergyTablePJ,
) -> dict[str, float | str]:
    workload = mlp_workload(
        input_dim,
        hidden_dim,
        num_classes,
        replace_linear=approx_linear,
        replace_gelu=approx_gelu,
    )
    return estimate_workload_compute_energy(
        workload,
        BPLAComputeConfig(affine_path, dyadic_terms),
        energy_table,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pretrained MNIST MLP with B-PLA primitives.")
    parser.add_argument("--checkpoint", default=str(REFERENCE_ROOT / "test_model" / "mnist_mlp.pth"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-test-samples", type=int, default=1000)
    parser.add_argument("--linear-prefix-bits", type=int, default=4)
    parser.add_argument("--activation-prefix-bits", type=int, default=4)
    parser.add_argument("--gelu-range", type=float, default=4.0)
    parser.add_argument("--affine-path", choices=["float", "dyadic"], default="float")
    parser.add_argument("--dyadic-terms", type=int, default=1)
    parser.add_argument("--max-shift", type=int, default=16)
    parser.add_argument("--no-bpla-linear", action="store_true")
    parser.add_argument("--no-bpla-gelu", action="store_true")
    parser.add_argument("--energy-shift-pj", type=float, default=0.0)
    parser.add_argument("--energy-control-pj", type=float, default=0.005)
    parser.add_argument("--energy-tanh-pj", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)

    ann_model = ToyTransformerMLP(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_classes=checkpoint["num_classes"],
    ).to(device)
    ann_model.load_state_dict(checkpoint["state_dict"])
    ann_model.eval()

    loader = load_mnist(args.batch_size, args.max_test_samples)
    ann_result = evaluate(ann_model, loader, device)

    approx_linear = not args.no_bpla_linear
    approx_gelu = not args.no_bpla_gelu
    bpla_model = BPLAMLP(
        ann_model,
        linear_prefix_bits=args.linear_prefix_bits,
        activation_prefix_bits=args.activation_prefix_bits,
        gelu_range=args.gelu_range,
        affine_path=args.affine_path,
        dyadic_terms=args.dyadic_terms,
        max_shift=args.max_shift,
        approx_linear=approx_linear,
        approx_gelu=approx_gelu,
    ).to(device)
    bpla_result = evaluate(bpla_model, loader, device)

    energy = estimate_energy(
        checkpoint["input_dim"],
        checkpoint["hidden_dim"],
        checkpoint["num_classes"],
        approx_linear,
        approx_gelu,
        args.affine_path,
        args.dyadic_terms,
        ComputeEnergyTablePJ(
            fixed_shift=args.energy_shift_pj,
            small_control=args.energy_control_pj,
            fp32_tanh=args.energy_tanh_pj,
        ),
    )

    print("\nB-PLA MNIST MLP Experiment")
    print("=" * 72)
    print(f"samples                  : {ann_result.samples}")
    print(f"B-PLA linear             : {approx_linear} (prefix={args.linear_prefix_bits})")
    print(f"B-PLA GELU               : {approx_gelu} (prefix={args.activation_prefix_bits}, range=+/-{args.gelu_range})")
    print(f"affine path              : {args.affine_path} (terms={args.dyadic_terms}, max_shift={args.max_shift})")
    print(f"ANN accuracy             : {ann_result.accuracy:.2f}%")
    print(f"B-PLA accuracy           : {bpla_result.accuracy:.2f}%")
    print(f"accuracy drop            : {ann_result.accuracy - bpla_result.accuracy:.2f}%")
    print("-" * 72)
    print("Energy is compute-only theory, not a synthesis result.")
    print("\n" + format_compute_energy_report(energy))


if __name__ == "__main__":
    main()
