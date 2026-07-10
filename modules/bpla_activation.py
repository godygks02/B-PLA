"""
B-PLA Activation
================

Minimal Bit-Prefix Piecewise Linear Approximation for activation functions.

The activation frontend routes FP32 inputs by direct bit-field extraction: sign,
exponent, and mantissa prefix bits select a local affine segment. This avoids
modeling activation routing as an expensive runtime FP-to-fixed normalization
step while keeping the public method name as B-PLA activation.

Each selected segment approximates:

    f(x) ~= slope_i*x + intercept_i
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from modules.dyadic import DyadicTerms, dyadic_constant, quantize_signed_pot, shift_add_multiply


TARGETS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "relu": lambda x: np.maximum(x, 0.0),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "tanh": np.tanh,
    "gelu": lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))),
    "quick_gelu": lambda x: x / (1.0 + np.exp(-1.702 * x)),
}


@dataclass(frozen=True)
class BPLAActivationTable:
    slopes: np.ndarray
    intercepts: np.ndarray
    x_min: float
    x_max: float
    prefix_bits: int
    target_name: str
    min_e_routing: int
    max_e_routing: int


@dataclass(frozen=True)
class BPLADyadicActivationTable:
    slopes: DyadicTerms
    intercepts: DyadicTerms
    x_min: float
    x_max: float
    prefix_bits: int
    target_name: str
    min_e_routing: int
    max_e_routing: int


def build_activation_table(
    target_name: str = "gelu",
    prefix_bits: int = 4,
    x_min: float = -4.0,
    x_max: float = 4.0,
    samples_per_segment: int = 64,
    min_e_routing: int = -5,
) -> BPLAActivationTable:
    if target_name not in TARGETS:
        raise ValueError(f"Unknown target_name: {target_name}. Choose one of {sorted(TARGETS)}.")
    if prefix_bits <= 0 or prefix_bits > 10:
        raise ValueError("prefix_bits must be in [1, 10].")
    if not x_min < x_max:
        raise ValueError("x_min must be smaller than x_max.")
    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be at least 2.")

    target = TARGETS[target_name]
    max_abs = max(abs(float(x_min)), abs(float(x_max)), np.finfo(np.float32).tiny)
    max_e_routing = int(np.floor(np.log2(max_abs)))
    segments = _segment_count(prefix_bits, min_e_routing, max_e_routing)
    slopes = np.zeros(segments, dtype=np.float64)
    intercepts = np.zeros(segments, dtype=np.float64)
    grid_points = max(segments * samples_per_segment, 4096)
    xs_all = np.linspace(x_min, x_max, grid_points, dtype=np.float64)
    idx_all = _prefix_index_from_bits(
        xs_all,
        prefix_bits=prefix_bits,
        x_min=x_min,
        x_max=x_max,
        min_e_routing=min_e_routing,
        max_e_routing=max_e_routing,
    )

    for i in range(segments):
        xs = xs_all[idx_all == i]
        if xs.size >= 2:
            ys = target(xs)
            slopes[i], intercepts[i] = np.polyfit(xs, ys, deg=1)
        elif xs.size == 1:
            slopes[i] = 0.0
            intercepts[i] = target(xs)[0]
        else:
            # Empty bit-prefix cells can occur near range boundaries. They are
            # unreachable for the calibration range, so use a benign constant
            # fallback at zero.
            slopes[i] = 0.0
            intercepts[i] = target(np.array([0.0], dtype=np.float64))[0]

    return BPLAActivationTable(
        slopes=slopes,
        intercepts=intercepts,
        x_min=float(x_min),
        x_max=float(x_max),
        prefix_bits=prefix_bits,
        target_name=target_name,
        min_e_routing=int(min_e_routing),
        max_e_routing=int(max_e_routing),
    )


def build_dyadic_activation_table(
    target_name: str = "gelu",
    prefix_bits: int = 4,
    x_min: float = -4.0,
    x_max: float = 4.0,
    samples_per_segment: int = 64,
    min_e_routing: int = -5,
    slope_terms: int = 1,
    intercept_terms: int | None = None,
    max_shift: int = 16,
) -> BPLADyadicActivationTable:
    table = build_activation_table(
        target_name=target_name,
        prefix_bits=prefix_bits,
        x_min=x_min,
        x_max=x_max,
        samples_per_segment=samples_per_segment,
        min_e_routing=min_e_routing,
    )
    if intercept_terms is None:
        intercept_terms = slope_terms
    return BPLADyadicActivationTable(
        slopes=quantize_signed_pot(table.slopes, terms=slope_terms, max_shift=max_shift),
        intercepts=quantize_signed_pot(table.intercepts, terms=intercept_terms, max_shift=max_shift),
        x_min=table.x_min,
        x_max=table.x_max,
        prefix_bits=table.prefix_bits,
        target_name=table.target_name,
        min_e_routing=table.min_e_routing,
        max_e_routing=table.max_e_routing,
    )


def _prefix_index(x: np.ndarray, table: BPLAActivationTable) -> np.ndarray:
    return _prefix_index_from_bits(
        x,
        prefix_bits=table.prefix_bits,
        x_min=table.x_min,
        x_max=table.x_max,
        min_e_routing=table.min_e_routing,
        max_e_routing=table.max_e_routing,
    )


def _decompose_float32_bits(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x32 = np.asarray(x, dtype=np.float32)
    bits = x32.view(np.uint32)
    sign = (bits >> 31).astype(np.uint32)
    exponent_field = ((bits >> 23) & 0xFF).astype(np.int32)
    fraction_q23 = (bits & 0x7FFFFF).astype(np.uint32)
    normal = (exponent_field > 0) & (exponent_field < 0xFF)
    return sign, exponent_field - 127, fraction_q23, normal


def _segment_count(prefix_bits: int, min_e_routing: int, max_e_routing: int) -> int:
    exponent_bins = max_e_routing - min_e_routing + 1
    if exponent_bins <= 0:
        raise ValueError("max_e_routing must be greater than or equal to min_e_routing.")
    return 1 + 2 * exponent_bins * (1 << prefix_bits)


def _prefix_index_from_bits(
    x: np.ndarray,
    prefix_bits: int,
    x_min: float,
    x_max: float,
    min_e_routing: int,
    max_e_routing: int,
) -> np.ndarray:
    x_clip = np.clip(np.asarray(x, dtype=np.float64), x_min, x_max).astype(np.float32)
    sign, exponent, fraction_q23, normal = _decompose_float32_bits(x_clip)
    small_or_zero = (~normal) | (exponent < min_e_routing) | (x_clip == 0.0)

    exponent_bins = max_e_routing - min_e_routing + 1
    prefix = (fraction_q23 >> (23 - prefix_bits)).astype(np.int64)
    exp_bin = np.clip(exponent, min_e_routing, max_e_routing).astype(np.int64) - min_e_routing
    sign_bin = sign.astype(np.int64)

    idx = 1 + ((sign_bin * exponent_bins + exp_bin) << prefix_bits) + prefix
    return np.where(small_or_zero, 0, idx).astype(np.int64)


def bpla_activation(
    x: np.ndarray,
    target_name: str = "gelu",
    prefix_bits: int = 4,
    x_min: float = -4.0,
    x_max: float = 4.0,
    table: BPLAActivationTable | None = None,
    affine_path: str = "float",
    dyadic_table: BPLADyadicActivationTable | None = None,
    dyadic_terms: int = 1,
    intercept_terms: int | None = None,
    max_shift: int = 16,
    min_e_routing: int = -5,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    if affine_path == "float":
        if table is None:
            table = build_activation_table(
                target_name,
                prefix_bits,
                x_min,
                x_max,
                min_e_routing=min_e_routing,
            )
        idx = _prefix_index(x_arr, table)
        x_clip = np.clip(x_arr, table.x_min, table.x_max)
        return table.slopes[idx] * x_clip + table.intercepts[idx]
    if affine_path == "dyadic":
        if dyadic_table is None:
            dyadic_table = build_dyadic_activation_table(
                target_name=target_name,
                prefix_bits=prefix_bits,
                x_min=x_min,
                x_max=x_max,
                min_e_routing=min_e_routing,
                slope_terms=dyadic_terms,
                intercept_terms=intercept_terms,
                max_shift=max_shift,
            )
        elif dyadic_table.prefix_bits != prefix_bits:
            raise ValueError("dyadic_table.prefix_bits must match prefix_bits.")
        idx = _prefix_index_from_bits(
            x_arr,
            prefix_bits=dyadic_table.prefix_bits,
            x_min=dyadic_table.x_min,
            x_max=dyadic_table.x_max,
            min_e_routing=dyadic_table.min_e_routing,
            max_e_routing=dyadic_table.max_e_routing,
        )
        x_clip = np.clip(x_arr, dyadic_table.x_min, dyadic_table.x_max)
        slope_terms = DyadicTerms(dyadic_table.slopes.signs[idx], dyadic_table.slopes.shifts[idx])
        intercept_terms_selected = DyadicTerms(dyadic_table.intercepts.signs[idx], dyadic_table.intercepts.shifts[idx])
        return shift_add_multiply(x_clip, slope_terms) + dyadic_constant(intercept_terms_selected)
    raise ValueError("affine_path must be 'float' or 'dyadic'.")


def exact_activation(x: np.ndarray, target_name: str = "gelu") -> np.ndarray:
    if target_name not in TARGETS:
        raise ValueError(f"Unknown target_name: {target_name}. Choose one of {sorted(TARGETS)}.")
    return TARGETS[target_name](np.asarray(x, dtype=np.float64))


def error_summary(approx: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    approx = np.asarray(approx, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    abs_err = np.abs(approx - reference)
    return {
        "mae": float(abs_err.mean()),
        "max_abs": float(abs_err.max()),
        "rmse": float(np.sqrt(np.mean(abs_err**2))),
    }


if __name__ == "__main__":
    x = np.linspace(-4.0, 4.0, 10000)
    ref = exact_activation(x, "gelu")
    out = bpla_activation(x, "gelu", prefix_bits=4)
    print(error_summary(out, ref))
