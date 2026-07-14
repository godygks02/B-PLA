"""Term-free event-driven piecewise-linear spiking primitives.

The compiler in this module is allowed to use floating-point arithmetic while
building a coefficient table offline.  The runtime functions do not multiply
an input value by a PLA coefficient.  A bit-plane spike selects a precomputed
synaptic increment and the membrane only performs conditional additions.

For ``x = sign * sum_t spike[t] * 2**(p_t - F)`` and ``y = a*x + b``::

    increment[t] = quantize(a) * 2**(p_t - F)     # offline
    membrane = quantize(b)
    if spike[t]: membrane += sign * increment[t]  # runtime

This removes the signed-power-of-two term budget used by dyadic B-PLA.  The
remaining coefficient precision knob is an ordinary fixed-point word length.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SynapticFixedPointConfig:
    """Fixed-point format used for offline PLA coefficient compilation."""

    total_bits: int = 24
    fractional_bits: int = 18


@dataclass(frozen=True)
class CompiledAffineSynapses:
    """Per-route, per-bit-plane increments for an affine PLA table."""

    increments: np.ndarray
    bias: np.ndarray
    coefficient_scale: int


def _validate_synapse_config(config: SynapticFixedPointConfig) -> None:
    if config.total_bits < 2:
        raise ValueError("total_bits must be at least 2.")
    if config.fractional_bits < 0 or config.fractional_bits >= config.total_bits:
        raise ValueError("fractional_bits must be in [0, total_bits).")


def quantize_synaptic_coefficients(
    values: np.ndarray,
    config: SynapticFixedPointConfig,
) -> np.ndarray:
    """Quantize offline coefficients without a dyadic term expansion."""

    _validate_synapse_config(config)
    scale = 1 << config.fractional_bits
    q_min = -(1 << (config.total_bits - 1))
    q_max = (1 << (config.total_bits - 1)) - 1
    q = np.rint(np.asarray(values, dtype=np.float64) * float(scale)).astype(np.int64)
    q = np.clip(q, q_min, q_max)
    return q.astype(np.float64) / float(scale)


def compile_affine_synapses(
    slopes: np.ndarray,
    biases: np.ndarray,
    bit_positions: np.ndarray,
    input_fractional_bits: int,
    config: SynapticFixedPointConfig | None = None,
) -> CompiledAffineSynapses:
    """Compile affine coefficients into bit-plane synaptic increments.

    This function models an offline table-generation step.  ``np.ldexp`` maps
    to an exponent adjustment/constant shift and does not create a runtime
    coefficient-by-activation multiplier.
    """

    config = config or SynapticFixedPointConfig()
    slopes_q = quantize_synaptic_coefficients(slopes, config)
    biases_q = quantize_synaptic_coefficients(biases, config)
    shifts = np.asarray(bit_positions, dtype=np.int32) - int(input_fractional_bits)
    increments = np.ldexp(slopes_q[..., None], shifts)
    return CompiledAffineSynapses(
        increments=increments.astype(np.float64),
        bias=biases_q.astype(np.float64),
        coefficient_scale=1 << config.fractional_bits,
    )


def event_affine_accumulate(
    spikes: np.ndarray,
    signs: np.ndarray,
    selected_increments: np.ndarray,
    selected_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate selected affine synapses using conditional additions only.

    ``selected_increments`` must have the same shape as ``spikes``.  The
    returned event-add count is per output element and counts only active input
    spikes.  No activation/coefficient multiplication occurs in this runtime.
    """

    spike_arr = np.asarray(spikes, dtype=np.uint8)
    increments = np.asarray(selected_increments, dtype=np.float64)
    if spike_arr.shape != increments.shape:
        raise ValueError("spikes and selected_increments must have the same shape.")
    sign_arr = np.asarray(signs, dtype=np.int64)
    if sign_arr.shape != spike_arr.shape[:-1]:
        raise ValueError("signs must match spikes without the timestep dimension.")

    signed_increments = np.where(sign_arr[..., None] < 0, -increments, increments)
    membrane = np.asarray(selected_bias, dtype=np.float64).copy()
    event_adds = np.zeros(spike_arr.shape[:-1], dtype=np.int64)
    for timestep in range(spike_arr.shape[-1]):
        active = spike_arr[..., timestep].astype(bool)
        membrane += np.where(active, signed_increments[..., timestep], 0.0)
        event_adds += active.astype(np.int64)
    return membrane, event_adds


def event_dual_affine_accumulate(
    spikes_a: np.ndarray,
    spikes_b: np.ndarray,
    selected_increments_a: np.ndarray,
    selected_increments_b: np.ndarray,
    selected_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-input PLA runtime for ``a*x1 + b*x2 + c``."""

    ones_a = np.ones(np.asarray(spikes_a).shape[:-1], dtype=np.int64)
    first, adds_a = event_affine_accumulate(
        spikes_a,
        ones_a,
        selected_increments_a,
        selected_bias,
    )
    zeros = np.zeros_like(first, dtype=np.float64)
    second, adds_b = event_affine_accumulate(
        spikes_b,
        ones_a,
        selected_increments_b,
        zeros,
    )
    return first + second, adds_a + adds_b


def event_if_decode(
    spikes: np.ndarray,
    signs: np.ndarray,
    selected_increments: np.ndarray,
    selected_bias: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """IF decoding driven directly by precompiled synaptic increments."""

    if threshold <= 0.0:
        raise ValueError("threshold must be positive.")
    spike_arr = np.asarray(spikes, dtype=np.uint8)
    increments = np.asarray(selected_increments, dtype=np.float64)
    sign_arr = np.asarray(signs, dtype=np.int64)
    signed_increments = np.where(sign_arr[..., None] < 0, -increments, increments)
    membrane = np.asarray(selected_bias, dtype=np.float64).copy()
    pos_count = np.zeros(spike_arr.shape[:-1], dtype=np.int64)
    neg_count = np.zeros_like(pos_count)
    event_adds = np.zeros_like(pos_count)

    for timestep in range(spike_arr.shape[-1]):
        active = spike_arr[..., timestep].astype(bool)
        membrane += np.where(active, signed_increments[..., timestep], 0.0)
        event_adds += active.astype(np.int64)
        pos_fire = membrane >= threshold
        neg_fire = membrane <= -threshold
        if np.any(pos_fire):
            count = np.floor(membrane[pos_fire] / threshold).astype(np.int64)
            pos_count[pos_fire] += count
            membrane[pos_fire] -= count.astype(np.float64) * threshold
        if np.any(neg_fire):
            count = np.floor(-membrane[neg_fire] / threshold).astype(np.int64)
            neg_count[neg_fire] += count
            membrane[neg_fire] += count.astype(np.float64) * threshold

    decoded = (pos_count - neg_count).astype(np.float64) * threshold
    events = pos_count + neg_count
    return decoded, pos_count, neg_count, events, event_adds


def event_dual_if_decode(
    spikes_a: np.ndarray,
    spikes_b: np.ndarray,
    selected_increments_a: np.ndarray,
    selected_increments_b: np.ndarray,
    selected_bias: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IF runtime for a two-input affine PLA tile using event additions."""

    if threshold <= 0.0:
        raise ValueError("threshold must be positive.")
    spike_a = np.asarray(spikes_a, dtype=np.uint8)
    spike_b = np.asarray(spikes_b, dtype=np.uint8)
    increments_a = np.asarray(selected_increments_a, dtype=np.float64)
    increments_b = np.asarray(selected_increments_b, dtype=np.float64)
    if spike_a.shape != spike_b.shape or spike_a.shape != increments_a.shape or spike_b.shape != increments_b.shape:
        raise ValueError("dual IF spike and increment arrays must have equal shapes.")

    membrane = np.asarray(selected_bias, dtype=np.float64).copy()
    signed_count = np.zeros(spike_a.shape[:-1], dtype=np.int64)
    event_adds = np.zeros_like(signed_count)
    for timestep in range(spike_a.shape[-1]):
        active_a = spike_a[..., timestep].astype(bool)
        active_b = spike_b[..., timestep].astype(bool)
        membrane += np.where(active_a, increments_a[..., timestep], 0.0)
        membrane += np.where(active_b, increments_b[..., timestep], 0.0)
        event_adds += active_a.astype(np.int64) + active_b.astype(np.int64)
        pos_fire = membrane >= threshold
        neg_fire = membrane <= -threshold
        if np.any(pos_fire):
            count = np.floor(membrane[pos_fire] / threshold).astype(np.int64)
            signed_count[pos_fire] += count
            membrane[pos_fire] -= count.astype(np.float64) * threshold
        if np.any(neg_fire):
            count = np.floor(-membrane[neg_fire] / threshold).astype(np.int64)
            signed_count[neg_fire] -= count
            membrane[neg_fire] += count.astype(np.float64) * threshold

    decoded = signed_count.astype(np.float64) * threshold
    return decoded, np.abs(signed_count).astype(np.int64), event_adds
