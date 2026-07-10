"""
B-PLA-SNN Multiplier
====================

Spiking reinterpretation of the B-PLA mantissa interaction approximation.

The FP32 sign and exponent paths remain deterministic conversion logic. The
nonlinear mantissa interaction term is evaluated as a prefix-routed PLA spiking
operator over fixed-point bit-plane spike streams. The default FS mode means
Few-Spikes, using coarse-to-fine spike timing rather than rate-style spike count.
The FS path uses a PAM-inspired progressive level approximation: each timestep
adds a residual correction between a coarser and a finer mantissa partition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from modules import bpla_multiplier as _bpla_multiplier


NeuronType = Literal["fs", "if"]


@dataclass(frozen=True)
class BPLASpikingMultiplierConfig:
    mantissa_bits: int = 12
    prefix_bits: int = 4
    neuron_type: NeuronType = "fs"
    threshold: float = 1.0 / 4096.0
    progressive_levels: bool = True


def _validate_config(config: BPLASpikingMultiplierConfig) -> None:
    if config.mantissa_bits <= 1:
        raise ValueError("mantissa_bits must be greater than 1.")
    if config.prefix_bits <= 0:
        raise ValueError("prefix_bits must be positive.")
    if config.prefix_bits >= config.mantissa_bits:
        raise ValueError("prefix_bits must be smaller than mantissa_bits.")
    if config.threshold <= 0.0:
        raise ValueError("threshold must be positive.")
    if config.neuron_type not in ("fs", "if"):
        raise ValueError("neuron_type must be 'fs' or 'if'.")


def encode_mantissa_bitplanes(
    fraction_q23: np.ndarray,
    mantissa_bits: int = 12,
) -> dict[str, np.ndarray]:
    """Encode FP32 fraction bits into mantissa bit-plane spikes."""

    if mantissa_bits <= 1 or mantissa_bits > 23:
        raise ValueError("mantissa_bits must be in [2, 23].")
    fraction_q23 = np.asarray(fraction_q23, dtype=np.uint32)
    frac_bits = mantissa_bits
    shift = 23 - frac_bits
    rounded = ((fraction_q23.astype(np.uint64) + (1 << max(0, shift - 1))) >> shift).astype(np.uint64)
    rounded = np.minimum(rounded, (1 << frac_bits) - 1)
    bit_positions = np.arange(frac_bits - 1, -1, -1, dtype=np.uint64)
    spikes = ((rounded[..., None] >> bit_positions) & 1).astype(np.uint8)
    weights = np.exp2(bit_positions.astype(np.float64) - frac_bits)
    decoded = np.sum(spikes.astype(np.float64) * weights, axis=-1)
    return {
        "spikes": spikes,
        "q_fraction": rounded.astype(np.uint64),
        "weights": weights,
        "decoded": decoded.astype(np.float64),
        "bit_positions": bit_positions,
    }


def _prefix_index_from_quantized(q_fraction: np.ndarray, config: BPLASpikingMultiplierConfig) -> np.ndarray:
    shift = config.mantissa_bits - config.prefix_bits
    return (np.asarray(q_fraction, dtype=np.uint64) >> shift).astype(np.int64)


def _prefix_index_for_level(q_fraction: np.ndarray, mantissa_bits: int, level: int) -> np.ndarray:
    shift = mantissa_bits - level
    return (np.asarray(q_fraction, dtype=np.uint64) >> shift).astype(np.int64)


def _emit_few_spikes(value: np.ndarray, threshold: float, timesteps: int) -> tuple[np.ndarray, np.ndarray]:
    residual = np.asarray(value, dtype=np.float64).copy()
    decoded = np.zeros_like(residual, dtype=np.float64)
    events = np.zeros_like(residual, dtype=np.int64)
    signs = np.sign(residual)
    residual_mag = np.abs(residual)
    weights = threshold * np.exp2(np.arange(timesteps - 1, -1, -1, dtype=np.float64))
    for weight in weights:
        fire = residual_mag >= weight
        events += fire.astype(np.int64)
        decoded += fire.astype(np.float64) * weight * signs
        residual_mag -= fire.astype(np.float64) * weight
    return decoded, events


def _pla_cross_for_level(
    mant_a: np.ndarray,
    mant_b: np.ndarray,
    q_a: np.ndarray,
    q_b: np.ndarray,
    mantissa_bits: int,
    level: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PAM-style level-l local affine approximation to m_a*m_b."""

    idx_a = _prefix_index_for_level(q_a, mantissa_bits, level)
    idx_b = _prefix_index_for_level(q_b, mantissa_bits, level)
    segments = 1 << level
    centers = (np.arange(segments, dtype=np.float64) + 0.5) / float(segments)
    mu = centers[idx_a]
    nu = centers[idx_b]
    cross = nu * mant_a + mu * mant_b - mu * nu
    return cross, idx_a, idx_b


def _progressive_cross(
    mant_a: np.ndarray,
    mant_b: np.ndarray,
    q_a: np.ndarray,
    q_b: np.ndarray,
    config: BPLASpikingMultiplierConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate f_l - f_{l-1} residuals over PAM-style levels."""

    previous = np.zeros_like(mant_a, dtype=np.float64)
    membrane = np.zeros_like(mant_a, dtype=np.float64)
    residual_abs_sum = np.zeros_like(mant_a, dtype=np.float64)
    level_pairs = []
    level_values = []
    for level in range(1, config.prefix_bits + 1):
        current, idx_a, idx_b = _pla_cross_for_level(
            mant_a,
            mant_b,
            q_a,
            q_b,
            config.mantissa_bits,
            level,
        )
        residual = current - previous
        membrane += residual
        residual_abs_sum += np.abs(residual)
        previous = current
        level_pairs.append(np.stack([idx_a, idx_b], axis=-1))
        level_values.append(membrane.copy())
    return membrane, level_pairs[-1], np.stack(level_pairs, axis=-2), residual_abs_sum, np.stack(level_values, axis=-1)


def _if_accumulate(
    spikes_a: np.ndarray,
    spikes_b: np.ndarray,
    weights: np.ndarray,
    coeff_a: np.ndarray,
    coeff_b: np.ndarray,
    coeff_c: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    membrane = np.zeros(coeff_a.shape, dtype=np.float64)
    signed_count = np.zeros(coeff_a.shape, dtype=np.int64)
    timesteps = spikes_a.shape[-1]
    for t in range(timesteps):
        membrane += coeff_a * spikes_a[..., t] * weights[t]
        membrane += coeff_b * spikes_b[..., t] * weights[t]
        membrane += coeff_c / float(timesteps)
        pos_fire = membrane >= threshold
        neg_fire = membrane <= -threshold
        if np.any(pos_fire):
            n_fire = np.floor(membrane[pos_fire] / threshold).astype(np.int64)
            signed_count[pos_fire] += n_fire
            membrane[pos_fire] -= n_fire.astype(np.float64) * threshold
        if np.any(neg_fire):
            n_fire = np.floor(-membrane[neg_fire] / threshold).astype(np.int64)
            signed_count[neg_fire] -= n_fire
            membrane[neg_fire] += n_fire.astype(np.float64) * threshold
    decoded = signed_count.astype(np.float64) * threshold
    return decoded, np.abs(signed_count).astype(np.int64)


def bpla_snn_multiply(
    a: np.ndarray,
    b: np.ndarray,
    config: BPLASpikingMultiplierConfig | None = None,
    coeffs: _bpla_multiplier.BPLACoefficients | None = None,
) -> dict[str, np.ndarray | dict[str, float]]:
    """Approximate FP32 multiplication with a spiking B-PLA mantissa core."""

    if config is None:
        config = BPLASpikingMultiplierConfig()
    _validate_config(config)

    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)
    pa = _bpla_multiplier.decompose_float32(a32)
    pb = _bpla_multiplier.decompose_float32(b32)
    enc_a = encode_mantissa_bitplanes(pa.fraction_q23, config.mantissa_bits)
    enc_b = encode_mantissa_bitplanes(pb.fraction_q23, config.mantissa_bits)

    idx_a = _prefix_index_from_quantized(enc_a["q_fraction"], config)
    idx_b = _prefix_index_from_quantized(enc_b["q_fraction"], config)
    if coeffs is None:
        coeffs = _bpla_multiplier.build_bpla_coefficients(config.prefix_bits)
    elif coeffs.prefix_bits != config.prefix_bits:
        raise ValueError("coeffs.prefix_bits must match config.prefix_bits.")

    mant_a = enc_a["decoded"]
    mant_b = enc_b["decoded"]

    if config.neuron_type == "fs":
        if config.progressive_levels:
            cross_raw, final_pairs, level_pairs, residual_abs_sum, level_values = _progressive_cross(
                mant_a,
                mant_b,
                enc_a["q_fraction"],
                enc_b["q_fraction"],
                config,
            )
            idx_a = final_pairs[..., 0]
            idx_b = final_pairs[..., 1]
        else:
            ca = coeffs.a[idx_a, idx_b]
            cb = coeffs.b[idx_a, idx_b]
            cc = coeffs.c[idx_a, idx_b]
            cross_raw = ca * mant_a + cb * mant_b + cc
            level_pairs = np.stack([idx_a, idx_b], axis=-1)[..., None, :]
            residual_abs_sum = np.abs(cross_raw)
            level_values = cross_raw[..., None]
        cross, cross_events = _emit_few_spikes(cross_raw, config.threshold, config.mantissa_bits)
    else:
        ca = coeffs.a[idx_a, idx_b]
        cb = coeffs.b[idx_a, idx_b]
        cc = coeffs.c[idx_a, idx_b]
        cross, cross_events = _if_accumulate(
            enc_a["spikes"],
            enc_b["spikes"],
            enc_a["weights"],
            ca,
            cb,
            cc,
            config.threshold,
        )
        level_pairs = np.stack([idx_a, idx_b], axis=-1)[..., None, :]
        residual_abs_sum = np.abs(cross)
        level_values = cross[..., None]

    sign = pa.sign ^ pb.sign
    valid = pa.normal & pb.normal
    mantissa = 1.0 + mant_a + mant_b + cross
    decoded = _bpla_multiplier._assemble_product(
        sign,
        pa.exponent + pb.exponent,
        mantissa,
        valid,
        _bpla_multiplier.exact_multiply(a32, b32),
    )

    input_events = np.sum(enc_a["spikes"], axis=-1) + np.sum(enc_b["spikes"], axis=-1)
    ops = {
        "lut_reads": float(np.size(decoded)),
        "spike_events": float(np.sum(input_events + cross_events)),
        "accumulate_ops": float(np.sum(input_events) * 2 + config.prefix_bits * np.size(decoded)),
        "threshold_compares": float(np.size(decoded) * config.mantissa_bits),
    }
    return {
        "decoded": decoded.astype(np.float64),
        "spike_events": (input_events + cross_events).astype(np.int64),
        "lut_reads": np.ones_like(decoded, dtype=np.int64),
        "accumulate_ops": (input_events * 2 + 1).astype(np.int64),
        "prefix_indices": np.stack([idx_a, idx_b], axis=-1),
        "level_indices": level_pairs,
        "level_values": level_values,
        "level_residual_abs_sum": residual_abs_sum,
        "ops": ops,
    }


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
