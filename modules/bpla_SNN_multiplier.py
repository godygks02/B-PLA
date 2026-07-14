"""
B-PLA-SNN Multiplier
====================

Spiking reinterpretation of the B-PLA mantissa interaction approximation.

The FP32 sign and exponent paths remain deterministic conversion logic. The
nonlinear mantissa interaction term is evaluated as a prefix-routed PLA spiking
operator over fixed-point bit-plane streams. Tile coefficients are compiled
offline into synaptic increments; runtime multiplication is replaced by binary
event-gated accumulation. The default FS mode means Few-Spikes rather than
rate-style spike count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from modules import bpla_multiplier as _bpla_multiplier
from modules import pla_snn


NeuronType = Literal["fs", "if"]


@dataclass(frozen=True)
class BPLASpikingMultiplierConfig:
    mantissa_bits: int = 12
    prefix_bits: int = 4
    neuron_type: NeuronType = "fs"
    threshold: float = 1.0 / 4096.0
    coefficient_bits: int = 24
    coefficient_fractional_bits: int = 18


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
    decoded = np.sum(np.where(spikes.astype(bool), weights, 0.0), axis=-1)
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
        signed_weight = np.where(signs < 0, -weight, weight)
        decoded += np.where(fire, signed_weight, 0.0)
        residual_mag -= np.where(fire, weight, 0.0)
    return decoded, events


def _synapse_config(config: BPLASpikingMultiplierConfig) -> pla_snn.SynapticFixedPointConfig:
    return pla_snn.SynapticFixedPointConfig(
        total_bits=config.coefficient_bits,
        fractional_bits=config.coefficient_fractional_bits,
    )


def _compile_selected_tile_synapses(
    coeffs: _bpla_multiplier.BPLACoefficients,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    bit_positions: np.ndarray,
    config: BPLASpikingMultiplierConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zeros = np.zeros_like(coeffs.c, dtype=np.float64)
    compiled_a = pla_snn.compile_affine_synapses(
        coeffs.a,
        coeffs.c,
        bit_positions,
        config.mantissa_bits,
        _synapse_config(config),
    )
    compiled_b = pla_snn.compile_affine_synapses(
        coeffs.b,
        zeros,
        bit_positions,
        config.mantissa_bits,
        _synapse_config(config),
    )
    return (
        compiled_a.increments[idx_a, idx_b],
        compiled_b.increments[idx_a, idx_b],
        compiled_a.bias[idx_a, idx_b],
    )


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
        increments_a, increments_b, bias = _compile_selected_tile_synapses(
            coeffs, idx_a, idx_b, enc_a["bit_positions"], config
        )
        cross_raw, input_event_adds = pla_snn.event_dual_affine_accumulate(
            enc_a["spikes"], enc_b["spikes"], increments_a, increments_b, bias
        )
        cross, cross_events = _emit_few_spikes(cross_raw, config.threshold, config.mantissa_bits)
    else:
        increments_a, increments_b, bias = _compile_selected_tile_synapses(
            coeffs, idx_a, idx_b, enc_a["bit_positions"], config
        )
        cross, cross_events, input_event_adds = pla_snn.event_dual_if_decode(
            enc_a["spikes"],
            enc_b["spikes"],
            increments_a,
            increments_b,
            bias,
            config.threshold,
        )

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
        "accumulate_ops": float(np.sum(input_event_adds) + config.prefix_bits * np.size(decoded)),
        "threshold_compares": float(np.size(decoded) * config.mantissa_bits),
        "runtime_multiplications": 0.0,
    }
    return {
        "decoded": decoded.astype(np.float64),
        "spike_events": (input_events + cross_events).astype(np.int64),
        "lut_reads": np.ones_like(decoded, dtype=np.int64),
        "accumulate_ops": (input_event_adds + 1).astype(np.int64),
        "prefix_indices": np.stack([idx_a, idx_b], axis=-1),
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
