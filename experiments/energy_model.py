"""
Operation-level energy model for a B-PLA multiplier.

This is not a Python runtime profiler. It estimates the energy of the intended
hardware datapath: FP split -> prefix/LUT -> fixed-point shift-add mantissa
evaluation -> normalization/packing.

Default energies are deliberately exposed as assumptions. They should be
replaced by values from a target technology library, synthesis report, or a
chosen reference table before making a publication-level claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import argparse

import numpy as np


@dataclass(frozen=True)
class EnergyTablePJ:
    """Per-operation energy assumptions in pJ."""

    int32_add: float = 0.10
    fp32_mul: float = 3.70
    fp32_add: float = 0.90
    lut_read: float = 0.04
    small_control: float = 0.005

    def fixed_add(self, bits: int) -> float:
        return self.int32_add * bits / 32.0


@dataclass(frozen=True)
class BPLAModel:
    """Datapath assumptions for one B-PLA multiply."""

    mantissa_bits: int = 24
    prefix_bits: int = 4
    terms_per_linear_coeff: int = 1
    offset_terms: int = 1
    count_shift_energy: bool = False
    variable_shift_mux_energy_pj: float = 0.0
    include_lut_read: bool = True
    include_pack_control: bool = True

    @property
    def linear_shift_terms(self) -> int:
        return 2 * self.terms_per_linear_coeff

    @property
    def total_affine_terms(self) -> int:
        return self.linear_shift_terms + self.offset_terms

    @property
    def affine_adds(self) -> int:
        return max(0, self.total_affine_terms - 1)

    @property
    def reconstruction_adds(self) -> int:
        # 1 + M1 + M2 + approx(M1M2)
        return 3

    @property
    def exponent_adds_equiv32(self) -> float:
        # E1 + E2 plus optional normalization increment.
        return 8 / 32 + 8 / 32

    @property
    def fixed_add_count(self) -> int:
        return self.affine_adds + self.reconstruction_adds


@dataclass(frozen=True)
class AccuracyConfig:
    samples: int = 100_000
    seed: int = 7
    exponent_min: int = -8
    exponent_max: int = 8
    max_shift: int = 16


def estimate_bpla_energy(model: BPLAModel, table: EnergyTablePJ) -> dict[str, float]:
    add_energy = model.fixed_add_count * table.fixed_add(model.mantissa_bits)
    exp_energy = model.exponent_adds_equiv32 * table.int32_add
    lut_energy = table.lut_read if model.include_lut_read else 0.0
    shift_energy = 0.0
    if model.count_shift_energy:
        shift_energy = model.linear_shift_terms * model.variable_shift_mux_energy_pj
    control_energy = table.small_control if model.include_pack_control else 0.0

    total = add_energy + exp_energy + lut_energy + shift_energy + control_energy
    return {
        "prefix_bits": model.prefix_bits,
        "mantissa_bits": model.mantissa_bits,
        "terms_per_linear_coeff": model.terms_per_linear_coeff,
        "lut_entries": float(1 << (2 * model.prefix_bits)),
        "linear_shift_terms": float(model.linear_shift_terms),
        "fixed_add_count": float(model.fixed_add_count),
        "fixed_add_energy_pj": add_energy,
        "exponent_energy_pj": exp_energy,
        "lut_energy_pj": lut_energy,
        "shift_mux_energy_pj": shift_energy,
        "control_energy_pj": control_energy,
        "total_bpla_pj": total,
        "fp32_mul_pj": table.fp32_mul,
        "bpla_over_fp32_mul": total / table.fp32_mul,
    }


def _signed_pot_approx(values: np.ndarray, terms: int, max_shift: int) -> np.ndarray:
    """Greedy signed power-of-two approximation for an array."""

    values = np.asarray(values, dtype=np.float64)
    approx = np.zeros_like(values)
    residual = values.copy()
    min_term = 2.0**-max_shift
    for _ in range(terms):
        active = np.abs(residual) >= 0.5 * min_term
        if not np.any(active):
            break
        shift = np.zeros_like(values, dtype=np.int32)
        shift[active] = np.rint(-np.log2(np.abs(residual[active]))).astype(np.int32)
        shift = np.clip(shift, 0, max_shift)
        term = np.sign(residual) * np.exp2(-shift)
        term = np.where(active, term, 0.0)
        approx += term
        residual = values - approx
    return approx


def _build_quantized_coefficients(model: BPLAModel, acc: AccuracyConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    segments = 1 << model.prefix_bits
    centers = (np.arange(segments, dtype=np.float64) + 0.5) / float(segments)
    mu = centers[:, None]
    nu = centers[None, :]

    a = np.broadcast_to(nu, (segments, segments)).copy()
    b = np.broadcast_to(mu, (segments, segments)).copy()
    c = -(mu @ nu)

    return (
        _signed_pot_approx(a, model.terms_per_linear_coeff, acc.max_shift),
        _signed_pot_approx(b, model.terms_per_linear_coeff, acc.max_shift),
        _signed_pot_approx(c, model.offset_terms, acc.max_shift),
    )


def _random_normal_fp32_pairs(acc: AccuracyConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(acc.seed)
    signs_a = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=acc.samples)
    signs_b = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=acc.samples)
    exp_a = rng.integers(acc.exponent_min, acc.exponent_max + 1, size=acc.samples, dtype=np.int32)
    exp_b = rng.integers(acc.exponent_min, acc.exponent_max + 1, size=acc.samples, dtype=np.int32)
    mant_a = 1.0 + rng.random(acc.samples, dtype=np.float64)
    mant_b = 1.0 + rng.random(acc.samples, dtype=np.float64)
    a = signs_a.astype(np.float64) * np.ldexp(mant_a, exp_a)
    b = signs_b.astype(np.float64) * np.ldexp(mant_b, exp_b)
    return a.astype(np.float32), b.astype(np.float32)


def _decompose_float32(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x32 = np.asarray(x, dtype=np.float32)
    bits = x32.view(np.uint32)
    sign = (bits >> 31).astype(np.uint32)
    exponent_field = ((bits >> 23) & 0xFF).astype(np.int32)
    fraction_q23 = (bits & 0x7FFFFF).astype(np.uint32)
    normal = (exponent_field > 0) & (exponent_field < 0xFF)
    return sign, exponent_field - 127, fraction_q23, normal


def bpla_multiply_hardware_like(a: np.ndarray, b: np.ndarray, model: BPLAModel, acc: AccuracyConfig) -> np.ndarray:
    if model.mantissa_bits < 2:
        raise ValueError("mantissa_bits must be at least 2.")
    if model.prefix_bits >= model.mantissa_bits:
        raise ValueError("prefix_bits must be smaller than mantissa_bits.")

    sign_a, exp_a, frac_a_q23, normal_a = _decompose_float32(a)
    sign_b, exp_b, frac_b_q23, normal_b = _decompose_float32(b)
    frac_bits = model.mantissa_bits - 1
    scale = float(1 << frac_bits)
    shift_down = 23 - frac_bits

    if shift_down >= 0:
        round_bias = 1 << max(0, shift_down - 1)
        frac_a_q = ((frac_a_q23.astype(np.uint64) + round_bias) >> shift_down).astype(np.uint64)
        frac_b_q = ((frac_b_q23.astype(np.uint64) + round_bias) >> shift_down).astype(np.uint64)
    else:
        frac_a_q = (frac_a_q23.astype(np.uint64) << (-shift_down)).astype(np.uint64)
        frac_b_q = (frac_b_q23.astype(np.uint64) << (-shift_down)).astype(np.uint64)

    frac_a_q = np.minimum(frac_a_q, (1 << frac_bits) - 1)
    frac_b_q = np.minimum(frac_b_q, (1 << frac_bits) - 1)
    mant_a = frac_a_q.astype(np.float64) / scale
    mant_b = frac_b_q.astype(np.float64) / scale

    idx_shift = frac_bits - model.prefix_bits
    idx_a = (frac_a_q >> idx_shift).astype(np.int64)
    idx_b = (frac_b_q >> idx_shift).astype(np.int64)
    coeff_a, coeff_b, coeff_c = _build_quantized_coefficients(model, acc)

    cross = coeff_a[idx_a, idx_b] * mant_a + coeff_b[idx_a, idx_b] * mant_b + coeff_c[idx_a, idx_b]
    mant = 1.0 + mant_a + mant_b + cross

    overflow = mant >= 2.0
    mant = np.where(overflow, mant * 0.5, mant)
    exp = exp_a + exp_b + overflow.astype(np.int32)
    mag = np.ldexp(mant, exp)
    signed = np.where((sign_a ^ sign_b) == 0, mag, -mag)
    fallback = np.asarray(a, dtype=np.float32).astype(np.float64) * np.asarray(b, dtype=np.float32).astype(np.float64)
    return np.where(normal_a & normal_b, signed, fallback).astype(np.float64)


def estimate_accuracy(model: BPLAModel, acc: AccuracyConfig) -> dict[str, float]:
    a, b = _random_normal_fp32_pairs(acc)
    approx = bpla_multiply_hardware_like(a, b, model, acc)
    reference = a.astype(np.float64) * b.astype(np.float64)
    finite = np.isfinite(reference)
    abs_err = np.abs(approx[finite] - reference[finite])
    rel_err = abs_err / (np.abs(reference[finite]) + 1e-30)
    return {
        "samples": float(finite.sum()),
        "mae": float(abs_err.mean()),
        "max_abs": float(abs_err.max()),
        "mean_rel": float(rel_err.mean()),
        "median_rel": float(np.quantile(rel_err, 0.50)),
        "p95_rel": float(np.quantile(rel_err, 0.95)),
        "p99_rel": float(np.quantile(rel_err, 0.99)),
        "max_rel": float(rel_err.max()),
    }


def estimate_bpla(model: BPLAModel, table: EnergyTablePJ, acc: AccuracyConfig) -> dict[str, float]:
    return estimate_bpla_energy(model, table) | estimate_accuracy(model, acc)


def print_report(name: str, model: BPLAModel, table: EnergyTablePJ, acc: AccuracyConfig | None = None) -> None:
    result = estimate_bpla_energy(model, table)
    print(f"\n{name}")
    print("-" * len(name))
    print(f"prefix bits          : {int(result['prefix_bits'])}")
    print(f"LUT entries          : {int(result['lut_entries'])}")
    print(f"linear shift terms   : {int(result['linear_shift_terms'])}")
    print(f"fixed-point adders   : {int(result['fixed_add_count'])}")
    print(f"fixed add energy     : {result['fixed_add_energy_pj']:.4f} pJ")
    print(f"exponent energy      : {result['exponent_energy_pj']:.4f} pJ")
    print(f"LUT energy           : {result['lut_energy_pj']:.4f} pJ")
    print(f"shift mux energy     : {result['shift_mux_energy_pj']:.4f} pJ")
    print(f"control energy       : {result['control_energy_pj']:.4f} pJ")
    print(f"B-PLA total          : {result['total_bpla_pj']:.4f} pJ")
    print(f"FP32 multiply        : {result['fp32_mul_pj']:.4f} pJ")
    print(f"B-PLA / FP32 mul     : {result['bpla_over_fp32_mul']:.3f}x")
    if acc is not None:
        accuracy = estimate_accuracy(model, acc)
        print(f"samples              : {int(accuracy['samples'])}")
        print(f"mean rel error       : {accuracy['mean_rel']:.6e}")
        print(f"p95 rel error        : {accuracy['p95_rel']:.6e}")
        print(f"p99 rel error        : {accuracy['p99_rel']:.6e}")
        print(f"max rel error        : {accuracy['max_rel']:.6e}")


def sweep_configs(
    table: EnergyTablePJ,
    acc: AccuracyConfig,
    prefix_bits_list: list[int],
    terms_list: list[int],
    mantissa_bits_list: list[int],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for prefix_bits in prefix_bits_list:
        for terms in terms_list:
            for mantissa_bits in mantissa_bits_list:
                if prefix_bits >= mantissa_bits:
                    continue
                model = BPLAModel(
                    mantissa_bits=mantissa_bits,
                    prefix_bits=prefix_bits,
                    terms_per_linear_coeff=terms,
                    offset_terms=terms,
                    include_lut_read=True,
                )
                rows.append(estimate_bpla(model, table, acc))
    return rows


def print_sweep(rows: list[dict[str, float]], limit: int = 20) -> None:
    rows = sorted(rows, key=lambda row: (row["p99_rel"], row["total_bpla_pj"]))
    header = (
        "prefix  terms  mant  LUT      adds  energy(pJ)  "
        "mean_rel    p99_rel     max_rel"
    )
    print("\nSweep results, sorted by p99 relative error")
    print(header)
    print("-" * len(header))
    for row in rows[:limit]:
        print(
            f"{int(row['prefix_bits']):>6}  "
            f"{int(row['terms_per_linear_coeff']):>5}  "
            f"{int(row['mantissa_bits']):>4}  "
            f"{int(row['lut_entries']):>7}  "
            f"{int(row['fixed_add_count']):>4}  "
            f"{row['total_bpla_pj']:>10.4f}  "
            f"{row['mean_rel']:.3e}  "
            f"{row['p99_rel']:.3e}  "
            f"{row['max_rel']:.3e}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate B-PLA multiplier energy and arithmetic accuracy.")
    parser.add_argument("--sweep", action="store_true", help="Run a prefix/term/mantissa-bit sweep.")
    parser.add_argument("--samples", type=int, default=100_000, help="Number of random FP32 input pairs.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--limit", type=int, default=20, help="Number of sweep rows to print.")
    parser.add_argument("--prefix-bits", type=int, default=8, help="Sweep prefix bit counts from 1 through this value.")
    parser.add_argument("--terms", type=int, default=3, help="Sweep term counts from 1 through this value.")
    parser.add_argument("--mantissa-bits", type=int, nargs="+", default=[12, 16, 20, 24])
    parser.add_argument("--max-shift", type=int, default=16, help="Largest right-shift used in coefficient quantization.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    default_table = EnergyTablePJ()
    accuracy_config = AccuracyConfig(samples=args.samples, seed=args.seed, max_shift=args.max_shift)

    if args.sweep:
        rows = sweep_configs(
            default_table,
            accuracy_config,
            prefix_bits_list=list(range(1, args.prefix_bits + 1)),
            terms_list=list(range(1, args.terms + 1)),
            mantissa_bits_list=args.mantissa_bits,
        )
        print_sweep(rows, limit=args.limit)
        raise SystemExit(0)

    print_report(
        "Compact 1-term dyadic model",
        BPLAModel(
            mantissa_bits=24,
            prefix_bits=4,
            terms_per_linear_coeff=1,
            offset_terms=1,
            include_lut_read=True,
        ),
        default_table,
        accuracy_config,
    )

    print_report(
        "2-term dyadic model",
        BPLAModel(
            mantissa_bits=24,
            prefix_bits=4,
            terms_per_linear_coeff=2,
            offset_terms=1,
            include_lut_read=True,
        ),
        default_table,
        accuracy_config,
    )

    print_report(
        "Low-precision 1-term model",
        BPLAModel(
            mantissa_bits=16,
            prefix_bits=4,
            terms_per_linear_coeff=1,
            offset_terms=1,
            include_lut_read=True,
        ),
        default_table,
        accuracy_config,
    )
