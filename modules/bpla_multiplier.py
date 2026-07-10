"""
B-PLA Multiplier
================

Minimal Bit-Prefix Piecewise Linear Approximation for FP32 multiplication.

The product is decomposed as:

    x1 * x2 = sign * 2^(e1 + e2) * (1 + m1 + m2 + m1*m2)

B-PLA approximates only the nonlinear mantissa interaction term:

    m1*m2 ~= a_ij*m1 + b_ij*m2 + c_ij

The tile index (i, j) is selected by the top prefix bits of m1 and m2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modules.dyadic import DyadicTerms, quantize_signed_pot, shift_add_multiply, dyadic_constant


@dataclass(frozen=True)
class FP32Parts:
    sign: np.ndarray
    exponent: np.ndarray
    fraction_q23: np.ndarray
    fraction: np.ndarray
    normal: np.ndarray


@dataclass(frozen=True)
class BPLACoefficients:
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    prefix_bits: int


@dataclass(frozen=True)
class BPLADyadicCoefficients:
    a: DyadicTerms
    b: DyadicTerms
    c: DyadicTerms
    prefix_bits: int


def decompose_float32(x: np.ndarray) -> FP32Parts:
    x32 = np.asarray(x, dtype=np.float32)
    bits = x32.view(np.uint32)
    sign = (bits >> 31).astype(np.uint32)
    exponent_field = ((bits >> 23) & 0xFF).astype(np.int32)
    fraction_q23 = (bits & 0x7FFFFF).astype(np.uint32)
    normal = (exponent_field > 0) & (exponent_field < 0xFF)
    return FP32Parts(
        sign=sign,
        exponent=exponent_field - 127,
        fraction_q23=fraction_q23,
        fraction=fraction_q23.astype(np.float64) / float(1 << 23),
        normal=normal,
    )


def exact_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=np.float32) * np.asarray(b, dtype=np.float32)).astype(np.float64)


def build_bpla_coefficients(prefix_bits: int = 4) -> BPLACoefficients:
    if prefix_bits <= 0 or prefix_bits > 10:
        raise ValueError("prefix_bits must be in [1, 10].")

    segments = 1 << prefix_bits
    centers = (np.arange(segments, dtype=np.float64) + 0.5) / float(segments)
    mu = centers[:, None]
    nu = centers[None, :]

    return BPLACoefficients(
        a=np.broadcast_to(nu, (segments, segments)).copy(),
        b=np.broadcast_to(mu, (segments, segments)).copy(),
        c=-(mu @ nu),
        prefix_bits=prefix_bits,
    )


def build_bpla_dyadic_coefficients(
    prefix_bits: int = 4,
    terms_per_linear_coeff: int = 1,
    offset_terms: int | None = None,
    max_shift: int = 16,
) -> BPLADyadicCoefficients:
    coeffs = build_bpla_coefficients(prefix_bits)
    if offset_terms is None:
        offset_terms = terms_per_linear_coeff
    return BPLADyadicCoefficients(
        a=quantize_signed_pot(coeffs.a, terms=terms_per_linear_coeff, max_shift=max_shift),
        b=quantize_signed_pot(coeffs.b, terms=terms_per_linear_coeff, max_shift=max_shift),
        c=quantize_signed_pot(coeffs.c, terms=offset_terms, max_shift=max_shift),
        prefix_bits=prefix_bits,
    )


def _prefix_index(fraction_q23: np.ndarray, prefix_bits: int) -> np.ndarray:
    return (fraction_q23 >> (23 - prefix_bits)).astype(np.int64)


def _assemble_product(
    sign: np.ndarray,
    exponent_sum: np.ndarray,
    mantissa: np.ndarray,
    valid: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    overflow = mantissa >= 2.0
    mantissa = np.where(overflow, mantissa * 0.5, mantissa)
    exponent = exponent_sum + overflow.astype(np.int32)
    magnitude = np.ldexp(mantissa, exponent.astype(np.int32))
    signed = np.where(sign == 0, magnitude, -magnitude)
    return np.where(valid, signed, fallback).astype(np.float64)


def bpla_multiply(
    a: np.ndarray,
    b: np.ndarray,
    prefix_bits: int = 4,
    coeffs: BPLACoefficients | None = None,
    affine_path: str = "float",
    dyadic_coeffs: BPLADyadicCoefficients | None = None,
    dyadic_terms: int = 1,
    max_shift: int = 16,
) -> np.ndarray:
    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)
    pa = decompose_float32(a32)
    pb = decompose_float32(b32)

    idx_a = _prefix_index(pa.fraction_q23, prefix_bits)
    idx_b = _prefix_index(pb.fraction_q23, prefix_bits)
    if affine_path == "float":
        if coeffs is None:
            coeffs = build_bpla_coefficients(prefix_bits)
        elif coeffs.prefix_bits != prefix_bits:
            raise ValueError("coeffs.prefix_bits must match prefix_bits.")
        cross = (
            coeffs.a[idx_a, idx_b] * pa.fraction
            + coeffs.b[idx_a, idx_b] * pb.fraction
            + coeffs.c[idx_a, idx_b]
        )
    elif affine_path == "dyadic":
        if dyadic_coeffs is None:
            dyadic_coeffs = build_bpla_dyadic_coefficients(
                prefix_bits=prefix_bits,
                terms_per_linear_coeff=dyadic_terms,
                offset_terms=dyadic_terms,
                max_shift=max_shift,
            )
        elif dyadic_coeffs.prefix_bits != prefix_bits:
            raise ValueError("dyadic_coeffs.prefix_bits must match prefix_bits.")
        a_terms = DyadicTerms(dyadic_coeffs.a.signs[idx_a, idx_b], dyadic_coeffs.a.shifts[idx_a, idx_b])
        b_terms = DyadicTerms(dyadic_coeffs.b.signs[idx_a, idx_b], dyadic_coeffs.b.shifts[idx_a, idx_b])
        c_terms = DyadicTerms(dyadic_coeffs.c.signs[idx_a, idx_b], dyadic_coeffs.c.shifts[idx_a, idx_b])
        cross = shift_add_multiply(pa.fraction, a_terms) + shift_add_multiply(pb.fraction, b_terms) + dyadic_constant(c_terms)
    else:
        raise ValueError("affine_path must be 'float' or 'dyadic'.")

    sign = pa.sign ^ pb.sign
    valid = pa.normal & pb.normal
    mantissa = 1.0 + pa.fraction + pb.fraction + cross
    return _assemble_product(sign, pa.exponent + pb.exponent, mantissa, valid, exact_multiply(a32, b32))


def error_summary(approx: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    approx = np.asarray(approx, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    finite = np.isfinite(reference)
    abs_err = np.abs(approx[finite] - reference[finite])
    rel_err = abs_err / (np.abs(reference[finite]) + 1e-30)
    return {
        "mae": float(abs_err.mean()),
        "max_abs": float(abs_err.max()),
        "mean_rel": float(rel_err.mean()),
        "p99_rel": float(np.quantile(rel_err, 0.99)),
        "max_rel": float(rel_err.max()),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.0, size=10000).astype(np.float32)
    y = rng.normal(0.0, 1.0, size=10000).astype(np.float32)
    ref = exact_multiply(x, y)
    out = bpla_multiply(x, y, prefix_bits=4)
    print(error_summary(out, ref))
