"""
B-PLA-SNN Activation
====================

Prefix-routed piecewise-linear spiking activation module.

This module reinterprets a B-PLA activation table as a bank of segment-specific
spiking neurons. PLA slopes are compiled offline into per-bit-plane synaptic
increments. At runtime, binary input events gate additions into the membrane;
there is no slope-by-activation multiplication and no dyadic term budget. The
default neuron emits an FS-style Few-Spikes output. An IF baseline is provided
for ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from modules import bpla_activation as _bpla_activation
from modules import pla_snn


NeuronType = Literal["fs", "if"]


@dataclass(frozen=True)
class BitPlaneEncodingConfig:
    bit_width: int = 12
    fractional_bits: int = 8
    signed: bool = True
    msb_first: bool = True


@dataclass(frozen=True)
class BPLASpikingNeuronConfig:
    neuron_type: NeuronType = "fs"
    threshold: float = 1.0 / 256.0
    bit_width: int = 12
    fractional_bits: int = 8
    prefix_bits: int = 4
    target_name: str = "gelu"
    x_min: float = -4.0
    x_max: float = 4.0
    min_e_routing: int = -5
    coefficient_bits: int = 24
    coefficient_fractional_bits: int = 18


def _validate_encoding_config(config: BitPlaneEncodingConfig) -> None:
    if config.bit_width <= 0:
        raise ValueError("bit_width must be positive.")
    if config.fractional_bits < 0:
        raise ValueError("fractional_bits must be non-negative.")
    if config.fractional_bits >= config.bit_width:
        raise ValueError("fractional_bits must be smaller than bit_width.")


def encode_bitplane_spikes(
    x: np.ndarray,
    config: BitPlaneEncodingConfig | None = None,
) -> dict[str, np.ndarray]:
    """Encode values into sign-magnitude fixed-point bit-plane spikes.

    Returns a dictionary with binary `spikes` ordered by timestep, the signed
    fixed-point integer, sign, magnitude integer, dyadic weights, and decoded
    value reconstructed from the bit-plane stream.
    """

    if config is None:
        config = BitPlaneEncodingConfig()
    _validate_encoding_config(config)

    x_arr = np.asarray(x, dtype=np.float64)
    scale = float(1 << config.fractional_bits)
    if config.signed:
        max_mag = (1 << (config.bit_width - 1)) - 1
    else:
        max_mag = (1 << config.bit_width) - 1

    q_signed = np.rint(x_arr * scale).astype(np.int64)
    q_signed = np.clip(q_signed, -max_mag if config.signed else 0, max_mag)
    sign = np.sign(q_signed).astype(np.int64)
    sign = np.where(sign == 0, 1, sign)
    magnitude = np.abs(q_signed).astype(np.int64)

    bit_positions = np.arange(config.bit_width - 1, -1, -1, dtype=np.int64)
    if not config.msb_first:
        bit_positions = bit_positions[::-1]
    spikes = ((magnitude[..., None] >> bit_positions) & 1).astype(np.uint8)
    weights = np.exp2(bit_positions.astype(np.float64) - config.fractional_bits)
    decoded_mag = np.sum(np.where(spikes.astype(bool), weights, 0.0), axis=-1)
    decoded = decoded_mag * sign

    return {
        "spikes": spikes,
        "q_signed": q_signed,
        "sign": sign,
        "magnitude": magnitude,
        "weights": weights,
        "decoded": decoded.astype(np.float64),
        "bit_positions": bit_positions,
    }


def prefix_index_from_bitplanes(spikes: np.ndarray, prefix_bits: int) -> np.ndarray:
    """Return a simple magnitude prefix from the first `prefix_bits` timesteps."""

    if prefix_bits <= 0:
        raise ValueError("prefix_bits must be positive.")
    spike_arr = np.asarray(spikes, dtype=np.uint8)
    if spike_arr.shape[-1] < prefix_bits:
        raise ValueError("spikes last dimension is smaller than prefix_bits.")
    prefix = spike_arr[..., :prefix_bits].astype(np.int64)
    powers = 1 << np.arange(prefix_bits - 1, -1, -1, dtype=np.int64)
    return np.sum(prefix * powers, axis=-1).astype(np.int64)


def _emit_signed_spikes(value: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if threshold <= 0.0:
        raise ValueError("threshold must be positive.")
    value = np.asarray(value, dtype=np.float64)
    pos = np.rint(np.maximum(value, 0.0) / threshold).astype(np.int64)
    neg = np.rint(np.maximum(-value, 0.0) / threshold).astype(np.int64)
    decoded = (pos - neg).astype(np.float64) * threshold
    return decoded, pos, neg


def _emit_few_spikes(
    value: np.ndarray,
    threshold: float,
    timesteps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode signed values with at most `timesteps` spikes per sign channel.

    This follows the FS-neuron spirit: spike timing carries coarse-to-fine
    value information. Each timestep has a decoding weight d(t), and a spike is
    emitted only when the residual membrane exceeds the current threshold.
    """

    if threshold <= 0.0:
        raise ValueError("threshold must be positive.")
    value = np.asarray(value, dtype=np.float64)
    residual_pos = np.maximum(value, 0.0).copy()
    residual_neg = np.maximum(-value, 0.0).copy()
    decoded_pos = np.zeros_like(value, dtype=np.float64)
    decoded_neg = np.zeros_like(value, dtype=np.float64)
    count_pos = np.zeros_like(value, dtype=np.int64)
    count_neg = np.zeros_like(value, dtype=np.int64)

    weights = threshold * np.exp2(np.arange(timesteps - 1, -1, -1, dtype=np.float64))
    for weight in weights:
        fire_pos = residual_pos >= weight
        fire_neg = residual_neg >= weight
        count_pos += fire_pos.astype(np.int64)
        count_neg += fire_neg.astype(np.int64)
        decoded_pos += np.where(fire_pos, weight, 0.0)
        decoded_neg += np.where(fire_neg, weight, 0.0)
        residual_pos -= np.where(fire_pos, weight, 0.0)
        residual_neg -= np.where(fire_neg, weight, 0.0)

    decoded = decoded_pos - decoded_neg
    events = count_pos + count_neg
    return decoded, count_pos, count_neg, events


def _synapse_config(config: BPLASpikingNeuronConfig) -> pla_snn.SynapticFixedPointConfig:
    return pla_snn.SynapticFixedPointConfig(
        total_bits=config.coefficient_bits,
        fractional_bits=config.coefficient_fractional_bits,
    )


def _compile_selected_synapses(
    table: _bpla_activation.BPLAActivationTable,
    table_indices: np.ndarray,
    encoded: dict[str, np.ndarray],
    config: BPLASpikingNeuronConfig,
) -> tuple[np.ndarray, np.ndarray]:
    compiled = pla_snn.compile_affine_synapses(
        table.slopes,
        table.intercepts,
        encoded["bit_positions"],
        config.fractional_bits,
        _synapse_config(config),
    )
    return compiled.increments[table_indices], compiled.bias[table_indices]


def bpla_snn_activation(
    x: np.ndarray,
    config: BPLASpikingNeuronConfig | None = None,
    table: _bpla_activation.BPLAActivationTable | None = None,
) -> dict[str, np.ndarray | dict[str, float]]:
    """Run prefix-routed B-PLA activation as a spiking neuron module."""

    if config is None:
        config = BPLASpikingNeuronConfig()
    if config.neuron_type not in ("fs", "if"):
        raise ValueError("neuron_type must be 'fs' or 'if'.")

    encoding_config = BitPlaneEncodingConfig(
        bit_width=config.bit_width,
        fractional_bits=config.fractional_bits,
        signed=True,
        msb_first=True,
    )
    encoded = encode_bitplane_spikes(np.clip(x, config.x_min, config.x_max), encoding_config)
    x_stream = encoded["decoded"]

    if table is None:
        table = _bpla_activation.build_activation_table(
            target_name=config.target_name,
            prefix_bits=config.prefix_bits,
            x_min=config.x_min,
            x_max=config.x_max,
            min_e_routing=config.min_e_routing,
        )

    table_indices = _bpla_activation._prefix_index(x_stream, table)
    increments, bias = _compile_selected_synapses(table, table_indices, encoded, config)

    if config.neuron_type == "fs":
        membrane, input_event_adds = pla_snn.event_affine_accumulate(
            encoded["spikes"],
            encoded["sign"],
            increments,
            bias,
        )
        decoded, pos, neg, events = _emit_few_spikes(membrane, config.threshold, config.bit_width)
    else:
        decoded, pos, neg, events, input_event_adds = pla_snn.event_if_decode(
            encoded["spikes"],
            encoded["sign"],
            increments,
            bias,
            config.threshold,
        )

    simple_prefix = prefix_index_from_bitplanes(encoded["spikes"], config.prefix_bits)
    ops = {
        "lut_reads": float(np.size(decoded)),
        "spike_events": float(np.sum(events)),
        "accumulate_ops": float(np.sum(input_event_adds) + config.prefix_bits * np.size(decoded)),
        "threshold_compares": float(np.size(decoded) * config.bit_width),
        "runtime_multiplications": 0.0,
    }
    return {
        "decoded": decoded.astype(np.float64),
        "spike_count_pos": pos,
        "spike_count_neg": neg,
        "spike_events": events,
        "prefix_indices": table_indices,
        "bitplane_prefix_indices": simple_prefix,
        "ops": ops,
    }


def error_summary(approx: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    approx = np.asarray(approx, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    abs_err = np.abs(approx - reference)
    return {
        "mae": float(abs_err.mean()),
        "max_abs": float(abs_err.max()),
        "rmse": float(np.sqrt(np.mean(abs_err**2))),
    }
