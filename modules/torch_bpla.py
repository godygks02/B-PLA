"""
CUDA-friendly B-PLA proxy operators for large-model sensitivity tests.

These operators are not a replacement for the hardware-faithful NumPy modules.
They avoid Python/NumPy round-trips so that pretrained PyTorch models can be
probed on CPU or CUDA. The goal is to answer: "Does the model tolerate this
class of B-PLA approximation?"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D


@dataclass(frozen=True)
class TorchBPLAConfig:
    prefix_bits: int = 4
    affine_path: str = "float"
    dyadic_terms: int = 2
    max_shift: int = 16
    activation_range: float = 4.0
    activation_samples_per_segment: int = 64
    linear_chunk_out: int = 32


def _validate_config(config: TorchBPLAConfig) -> None:
    if not 1 <= config.prefix_bits <= 10:
        raise ValueError("prefix_bits must be in [1, 10].")
    if config.affine_path not in {"float", "dyadic"}:
        raise ValueError("affine_path must be 'float' or 'dyadic'.")
    if config.dyadic_terms <= 0:
        raise ValueError("dyadic_terms must be positive.")
    if config.max_shift < 0:
        raise ValueError("max_shift must be non-negative.")


def _signed_pot_quantize(values: torch.Tensor, terms: int, max_shift: int) -> torch.Tensor:
    approx = torch.zeros_like(values)
    min_term = 2.0 ** -max_shift
    for _ in range(terms):
        residual = values - approx
        active = residual.abs() >= 0.5 * min_term
        shift = torch.round(-torch.log2(residual.abs().clamp_min(torch.finfo(values.dtype).tiny)))
        shift = shift.clamp(0, max_shift)
        term = residual.sign() * torch.pow(torch.tensor(2.0, device=values.device, dtype=values.dtype), -shift)
        approx = approx + torch.where(active, term, torch.zeros_like(term))
    return approx


def _maybe_dyadic(values: torch.Tensor, config: TorchBPLAConfig) -> torch.Tensor:
    if config.affine_path == "float":
        return values
    return _signed_pot_quantize(values, config.dyadic_terms, config.max_shift)


def _fraction_and_exponent(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    abs_x = x.abs()
    mant, exponent = torch.frexp(abs_x)
    normal = abs_x > 0
    fraction = torch.where(normal, mant * 2.0 - 1.0, torch.zeros_like(abs_x))
    unbiased_exponent = exponent - 1
    sign = torch.signbit(x)
    return fraction, unbiased_exponent, sign


def bpla_multiply_torch(a: torch.Tensor, b: torch.Tensor, config: TorchBPLAConfig) -> torch.Tensor:
    """Approximate elementwise multiplication with torch-native B-PLA logic."""

    _validate_config(config)
    dtype = torch.promote_types(a.dtype, b.dtype)
    a = a.to(dtype)
    b = b.to(dtype)

    frac_a, exp_a, sign_a = _fraction_and_exponent(a)
    frac_b, exp_b, sign_b = _fraction_and_exponent(b)
    segments = 1 << config.prefix_bits
    idx_a = torch.clamp((frac_a * segments).floor().to(torch.long), 0, segments - 1)
    idx_b = torch.clamp((frac_b * segments).floor().to(torch.long), 0, segments - 1)

    centers = (torch.arange(segments, device=a.device, dtype=dtype) + 0.5) / float(segments)
    mu = centers[idx_a]
    nu = centers[idx_b]
    coeff_a = _maybe_dyadic(nu, config)
    coeff_b = _maybe_dyadic(mu, config)
    coeff_c = _maybe_dyadic(-(mu * nu), config)

    cross = coeff_a * frac_a + coeff_b * frac_b + coeff_c
    mantissa = 1.0 + frac_a + frac_b + cross
    overflow = mantissa >= 2.0
    mantissa = torch.where(overflow, mantissa * 0.5, mantissa)
    exponent = exp_a + exp_b + overflow.to(exp_a.dtype)
    magnitude = torch.ldexp(mantissa, exponent)
    signed = torch.where(sign_a ^ sign_b, -magnitude, magnitude)
    return torch.where((a != 0) & (b != 0), signed, torch.zeros_like(signed))


def bpla_linear_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    config: TorchBPLAConfig,
) -> torch.Tensor:
    """Linear layer using B-PLA elementwise products, chunked over outputs."""

    _validate_config(config)
    original_shape = x.shape[:-1]
    x_flat = x.reshape(-1, x.shape[-1])
    rows = []
    chunk = max(1, config.linear_chunk_out)
    for start in range(0, weight.shape[0], chunk):
        w = weight[start : start + chunk]
        products = bpla_multiply_torch(x_flat[:, None, :], w[None, :, :], config)
        out = products.sum(dim=-1)
        if bias is not None:
            out = out + bias[start : start + chunk]
        rows.append(out)
    return torch.cat(rows, dim=-1).reshape(*original_shape, weight.shape[0])


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


TARGETS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "gelu": _gelu,
    "quick_gelu": lambda x: x * torch.sigmoid(1.702 * x),
    "relu": F.relu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
}


def build_activation_table_torch(
    target_name: str,
    config: TorchBPLAConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor | int | float]:
    if target_name not in TARGETS:
        raise ValueError(f"Unknown target_name: {target_name}.")
    _validate_config(config)

    x_min = -float(config.activation_range)
    x_max = float(config.activation_range)
    min_e_routing = -5
    max_e_routing = int(torch.floor(torch.log2(torch.tensor(max(abs(x_min), abs(x_max)), dtype=dtype))).item())
    exponent_bins = max_e_routing - min_e_routing + 1
    segments = 1 + 2 * exponent_bins * (1 << config.prefix_bits)

    xs = torch.linspace(x_min, x_max, max(segments * config.activation_samples_per_segment, 4096), device=device, dtype=dtype)
    idx = activation_prefix_index_torch(xs, config, min_e_routing, max_e_routing)
    ys = TARGETS[target_name](xs)
    slopes = torch.zeros(segments, device=device, dtype=dtype)
    intercepts = torch.zeros(segments, device=device, dtype=dtype)

    for seg in range(segments):
        mask = idx == seg
        x_seg = xs[mask]
        if x_seg.numel() >= 2:
            y_seg = ys[mask]
            x_mean = x_seg.mean()
            y_mean = y_seg.mean()
            denom = ((x_seg - x_mean) ** 2).sum().clamp_min(torch.finfo(dtype).eps)
            slope = ((x_seg - x_mean) * (y_seg - y_mean)).sum() / denom
            intercept = y_mean - slope * x_mean
            slopes[seg] = slope
            intercepts[seg] = intercept
        elif x_seg.numel() == 1:
            intercepts[seg] = TARGETS[target_name](x_seg)[0]

    slopes = _maybe_dyadic(slopes, config)
    intercepts = _maybe_dyadic(intercepts, config)
    return {
        "slopes": slopes,
        "intercepts": intercepts,
        "min_e_routing": min_e_routing,
        "max_e_routing": max_e_routing,
        "x_min": x_min,
        "x_max": x_max,
    }


def activation_prefix_index_torch(
    x: torch.Tensor,
    config: TorchBPLAConfig,
    min_e_routing: int,
    max_e_routing: int,
) -> torch.Tensor:
    x_clip = x.clamp(-float(config.activation_range), float(config.activation_range))
    fraction, exponent, sign = _fraction_and_exponent(x_clip)
    small_or_zero = (x_clip == 0) | (exponent < min_e_routing)
    exponent_bins = max_e_routing - min_e_routing + 1
    prefix = torch.clamp((fraction * (1 << config.prefix_bits)).floor().to(torch.long), 0, (1 << config.prefix_bits) - 1)
    exp_bin = exponent.clamp(min_e_routing, max_e_routing).to(torch.long) - min_e_routing
    sign_bin = sign.to(torch.long)
    idx = 1 + ((sign_bin * exponent_bins + exp_bin) << config.prefix_bits) + prefix
    return torch.where(small_or_zero, torch.zeros_like(idx), idx)


def bpla_activation_torch(x: torch.Tensor, table: dict[str, torch.Tensor | int | float], config: TorchBPLAConfig) -> torch.Tensor:
    idx = activation_prefix_index_torch(
        x,
        config,
        int(table["min_e_routing"]),
        int(table["max_e_routing"]),
    )
    x_clip = x.clamp(float(table["x_min"]), float(table["x_max"]))
    slopes = table["slopes"]
    intercepts = table["intercepts"]
    assert isinstance(slopes, torch.Tensor)
    assert isinstance(intercepts, torch.Tensor)
    return slopes[idx] * x_clip + intercepts[idx]


class TorchBPLALinear(nn.Module):
    def __init__(self, source: nn.Linear, config: TorchBPLAConfig):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        if source.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(source.bias.detach().clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return bpla_linear_torch(x, self.weight, self.bias, self.config)


class TorchBPLAActivation(nn.Module):
    def __init__(self, target_name: str = "gelu", config: TorchBPLAConfig | None = None):
        super().__init__()
        self.target_name = target_name
        self.config = config or TorchBPLAConfig()
        self._table: dict[str, torch.Tensor | int | float] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._table is None or self._table["slopes"].device != x.device or self._table["slopes"].dtype != x.dtype:
            self._table = build_activation_table_torch(self.target_name, self.config, x.device, x.dtype)
        return bpla_activation_torch(x, self._table, self.config)


class TorchBPLAConv1D(nn.Module):
    """B-PLA proxy replacement for HuggingFace GPT-style Conv1D."""

    def __init__(self, source: Conv1D, config: TorchBPLAConfig):
        super().__init__()
        self.config = config
        self.nf = source.nf
        self.weight = nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(source.bias.detach().clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        out = bpla_linear_torch(x, self.weight.t(), self.bias, self.config)
        return out.view(size_out)


def replace_linear_and_gelu(
    module: nn.Module,
    config: TorchBPLAConfig,
    replace_linear: bool = True,
    replace_gelu: bool = True,
    max_linear_modules: int | None = None,
) -> int:
    """In-place replacement helper for sensitivity checks on PyTorch models."""

    replaced_linear = 0
    for name, child in list(module.named_children()):
        if replace_linear and isinstance(child, nn.Linear) and (max_linear_modules is None or replaced_linear < max_linear_modules):
            setattr(module, name, TorchBPLALinear(child, config))
            replaced_linear += 1
            continue
        if replace_gelu and isinstance(child, nn.GELU):
            setattr(module, name, TorchBPLAActivation("gelu", config))
            continue
        replaced_linear += replace_linear_and_gelu(
            child,
            config=config,
            replace_linear=replace_linear,
            replace_gelu=replace_gelu,
            max_linear_modules=None if max_linear_modules is None else max_linear_modules - replaced_linear,
        )
    return replaced_linear


def replace_gpt2_conv1d_and_gelu(
    module: nn.Module,
    config: TorchBPLAConfig,
    replace_conv1d: bool = True,
    replace_gelu: bool = True,
    max_conv1d_modules: int | None = None,
) -> int:
    """In-place replacement helper for GPT-2 style models."""

    replaced_conv = 0
    for name, child in list(module.named_children()):
        if replace_conv1d and isinstance(child, Conv1D) and (max_conv1d_modules is None or replaced_conv < max_conv1d_modules):
            setattr(module, name, TorchBPLAConv1D(child, config))
            replaced_conv += 1
            continue
        if replace_gelu and isinstance(child, nn.GELU):
            setattr(module, name, TorchBPLAActivation("gelu", config))
            continue
        if replace_gelu and child.__class__.__name__.lower().endswith("geluactivation"):
            setattr(module, name, TorchBPLAActivation("gelu", config))
            continue
        replaced_conv += replace_gpt2_conv1d_and_gelu(
            child,
            config=config,
            replace_conv1d=replace_conv1d,
            replace_gelu=replace_gelu,
            max_conv1d_modules=None if max_conv1d_modules is None else max_conv1d_modules - replaced_conv,
        )
    return replaced_conv
