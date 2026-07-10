"""
MBE-style multiplication baseline
=================================

This module implements the floating-point multiplication rule described for
MBE neurons:

    x ~= sum_t d[t] s[t]
    x*y ~= sum_i sum_j d_x[i] d_y[j] s_x[i] s_y[j]

The original MBE paper learns exponential-decay neuron parameters. This file is
therefore a faithful structural baseline, not a reproduction of trained MBE
parameters. It uses a deterministic exponential/binary identity encoder so that
the multiplication cost can be compared with the prefix-routed B-PLA-SNN
multiplier under the same inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


@dataclass(frozen=True)
class MBEIdentityConfig:
    timesteps: int = 12
    value_range: float = 4.0
    exponential_decay: bool = True
    tau: float = 3.0


@dataclass(frozen=True)
class ExternalMBEConfig:
    """Configuration for using the user's PyTorch MBE implementation."""

    repo_path: str = r"C:\Users\cm120\Project\VLM_SNN_Research\MBE"
    checkpoint_path: str = r"C:\Users\cm120\Project\VLM_SNN_Research\MBE\mbe_models\activation\mbe_identity_T16_N4.pth"
    num_basis: int = 4
    timesteps: int = 16
    dt: float = 1.0
    alpha: float = 1.0
    device: str = "cpu"
    chunk_elements: int = 64


def _validate_config(config: MBEIdentityConfig) -> None:
    if config.timesteps <= 0:
        raise ValueError("timesteps must be positive.")
    if config.value_range <= 0.0:
        raise ValueError("value_range must be positive.")
    if config.tau <= 0.0:
        raise ValueError("tau must be positive.")


def _identity_weights(config: MBEIdentityConfig) -> np.ndarray:
    if config.exponential_decay:
        t = np.arange(config.timesteps, dtype=np.float64)
        weights = np.exp(-t / config.tau)
        weights = weights / np.sum(weights) * config.value_range
        return weights

    bit_positions = np.arange(config.timesteps - 1, -1, -1, dtype=np.float64)
    weights = np.exp2(bit_positions)
    weights = weights / np.sum(weights) * config.value_range
    return weights


def mbe_identity_encode(
    x: np.ndarray,
    config: MBEIdentityConfig | None = None,
) -> dict[str, np.ndarray]:
    """Encode values as signed weighted spike trains."""

    if config is None:
        config = MBEIdentityConfig()
    _validate_config(config)

    x64 = np.asarray(x, dtype=np.float64)
    signs = np.sign(x64)
    residual = np.minimum(np.abs(x64), config.value_range)
    weights = _identity_weights(config)
    spikes = np.zeros(x64.shape + (config.timesteps,), dtype=np.uint8)
    decoded_mag = np.zeros_like(x64, dtype=np.float64)

    for t, weight in enumerate(weights):
        fire = residual >= weight
        spikes[..., t] = fire.astype(np.uint8)
        decoded_mag += fire.astype(np.float64) * weight
        residual -= fire.astype(np.float64) * weight

    decoded = decoded_mag * signs
    clipped = np.clip(x64, -config.value_range, config.value_range)
    return {
        "spikes": spikes,
        "weights": weights,
        "signs": signs,
        "decoded": decoded,
        "clipped": clipped,
    }


def mbe_multiply(
    a: np.ndarray,
    b: np.ndarray,
    config: MBEIdentityConfig | None = None,
) -> dict[str, np.ndarray | dict[str, float]]:
    """Approximate multiplication with MBE-style spike outer products."""

    if config is None:
        config = MBEIdentityConfig()
    enc_a = mbe_identity_encode(a, config)
    enc_b = mbe_identity_encode(b, config)

    decoded = enc_a["decoded"] * enc_b["decoded"]
    events_a = np.sum(enc_a["spikes"], axis=-1)
    events_b = np.sum(enc_b["spikes"], axis=-1)
    outer_events = events_a * events_b
    ops = {
        "spike_events": float(np.sum(events_a + events_b)),
        "pairwise_spike_interactions": float(np.sum(outer_events)),
        "accumulate_ops": float(np.sum(outer_events)),
        "threshold_compares": float(np.size(decoded) * config.timesteps * 2),
        "lut_reads": 0.0,
    }
    return {
        "decoded": decoded.astype(np.float64),
        "input_decoded_a": enc_a["decoded"],
        "input_decoded_b": enc_b["decoded"],
        "spike_events": (events_a + events_b).astype(np.int64),
        "pairwise_spike_interactions": outer_events.astype(np.int64),
        "ops": ops,
    }


def external_mbe_multiply(
    a: np.ndarray,
    b: np.ndarray,
    config: ExternalMBEConfig | None = None,
) -> dict[str, np.ndarray | dict[str, float] | str]:
    """Approximate multiplication with the external PyTorch MBE implementation.

    This uses ``MBE/MBE_neurons.py`` and ``MBE/approximate_fp_mult.py`` when
    PyTorch is available. If the checkpoint exists, its trained identity MBE is
    loaded. Otherwise, an untrained MBE identity model is used.
    """

    if config is None:
        config = ExternalMBEConfig()

    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for external_mbe_multiply.") from exc

    repo_path = Path(config.repo_path)
    if not repo_path.exists():
        raise FileNotFoundError(f"External MBE repo not found: {repo_path}")
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    from MBE_neurons import MBENeuron
    from approximate_fp_mult import MBEMultiplier

    checkpoint_path = Path(config.checkpoint_path)
    if checkpoint_path.exists():
        mbe_id = MBENeuron.load(str(checkpoint_path), device=config.device)
        source = str(checkpoint_path)
    else:
        mbe_id = MBENeuron(
            num_basis=config.num_basis,
            timesteps=config.timesteps,
            dt=config.dt,
            alpha=config.alpha,
        ).to(config.device)
        source = "untrained_external_mbe"
    mbe_id.eval()
    multiplier = MBEMultiplier(mbe_id_model=mbe_id).to(config.device)
    multiplier.eval()

    a64 = np.asarray(a, dtype=np.float32)
    b64 = np.asarray(b, dtype=np.float32)
    if a64.shape != b64.shape:
        a64, b64 = np.broadcast_arrays(a64, b64)
    original_shape = a64.shape
    a_flat = a64.reshape(-1)
    b_flat = b64.reshape(-1)
    out_flat = np.empty_like(a_flat, dtype=np.float64)

    total_spike_events = 0.0
    total_pairwise = 0.0
    total_compares = 0.0
    timesteps = int(getattr(mbe_id, "timesteps", config.timesteps))
    num_basis = int(getattr(mbe_id, "num_basis", config.num_basis))
    k = timesteps * num_basis

    with torch.no_grad():
        for start in range(0, a_flat.size, config.chunk_elements):
            end = min(start + config.chunk_elements, a_flat.size)
            x1 = torch.from_numpy(a_flat[start:end]).to(config.device).view(-1, 1)
            x2 = torch.from_numpy(b_flat[start:end]).to(config.device).view(-1, 1)
            y = multiplier(x1, x2)
            out_flat[start:end] = y.detach().cpu().numpy().reshape(-1)

            _, s1, _ = mbe_id(x1, return_sequences=True)
            _, s2, _ = mbe_id(x2, return_sequences=True)
            ev1 = s1.reshape(k, -1).sum(dim=0)
            ev2 = s2.reshape(k, -1).sum(dim=0)
            total_spike_events += float((ev1 + ev2).sum().item())
            total_pairwise += float(getattr(multiplier, "last_interaction_sops", (ev1 * ev2).sum().item()))
            total_compares += float((end - start) * k * 2)

    ops = {
        "spike_events": total_spike_events,
        "pairwise_spike_interactions": total_pairwise,
        "accumulate_ops": total_pairwise,
        "threshold_compares": total_compares,
        "lut_reads": 0.0,
    }
    return {
        "decoded": out_flat.reshape(original_shape),
        "ops": ops,
        "source": source,
        "timesteps": np.array(timesteps),
        "num_basis": np.array(num_basis),
    }


def error_summary(approx: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    approx = np.asarray(approx, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    finite = np.isfinite(reference)
    abs_err = np.abs(approx[finite] - reference[finite])
    rel_err = abs_err / (np.abs(reference[finite]) + 1e-30)
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(abs_err**2))),
        "max_abs": float(abs_err.max()),
        "mean_rel": float(rel_err.mean()),
        "p99_rel": float(np.quantile(rel_err, 0.99)),
        "max_rel": float(rel_err.max()),
    }
