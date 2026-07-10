"""
Dyadic coefficient utilities for multiplierless B-PLA evaluation.

The functions here model a hardware-friendly signed power-of-two expansion:

    c ~= sum_t sign_t * 2^(-shift_t)

At runtime, multiplying an input by such a coefficient can be expressed as
shifted copies of the input followed by additions. In NumPy, `ldexp` is used as
the software analogue of a binary shift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DyadicTerms:
    signs: np.ndarray
    shifts: np.ndarray

    @property
    def terms(self) -> int:
        return int(self.signs.shape[-1])


def quantize_signed_pot(values: np.ndarray, terms: int = 1, max_shift: int = 16) -> DyadicTerms:
    """Greedy signed power-of-two approximation for scalar or array coefficients."""

    if terms <= 0:
        raise ValueError("terms must be positive.")
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative.")

    values = np.asarray(values, dtype=np.float64)
    signs = np.zeros(values.shape + (terms,), dtype=np.int8)
    shifts = np.zeros(values.shape + (terms,), dtype=np.int16)
    approx = np.zeros_like(values, dtype=np.float64)
    min_term = 2.0**-max_shift

    for term_idx in range(terms):
        residual = values - approx
        active = np.abs(residual) >= 0.5 * min_term
        if not np.any(active):
            break

        term_shift = np.zeros_like(values, dtype=np.int16)
        term_shift[active] = np.rint(-np.log2(np.abs(residual[active]))).astype(np.int16)
        term_shift = np.clip(term_shift, 0, max_shift).astype(np.int16)

        term_sign = np.where(active, np.sign(residual), 0.0).astype(np.int8)
        term_value = term_sign.astype(np.float64) * np.exp2(-term_shift.astype(np.int32))

        signs[..., term_idx] = term_sign
        shifts[..., term_idx] = term_shift
        approx = approx + term_value

    return DyadicTerms(signs=signs, shifts=shifts)


def terms_to_float(terms: DyadicTerms) -> np.ndarray:
    """Reconstruct dyadic coefficients as floating-point values for inspection."""

    return np.sum(terms.signs.astype(np.float64) * np.exp2(-terms.shifts.astype(np.int32)), axis=-1)


def shift_add_multiply(x: np.ndarray, terms: DyadicTerms) -> np.ndarray:
    """Evaluate x * dyadic_coefficient using shifted copies of x and addition."""

    x_arr = np.asarray(x, dtype=np.float64)
    expanded = np.expand_dims(x_arr, axis=-1)
    shifted = np.ldexp(expanded, -terms.shifts.astype(np.int32))
    signed = shifted * terms.signs.astype(np.float64)
    return np.sum(signed, axis=-1)


def dyadic_constant(terms: DyadicTerms) -> np.ndarray:
    """Evaluate a dyadic constant expansion."""

    return terms_to_float(terms)
