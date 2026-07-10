"""
External MBE activation baseline
================================

Adapter for the user's PyTorch MBE activation checkpoints under
``C:/Users/cm120/Project/VLM_SNN_Research/MBE``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


_DEFAULT_CHECKPOINTS = {
    "gelu": r"C:\Users\cm120\Project\VLM_SNN_Research\MBE\mbe_models\activation\mbe_gelu_model.pth",
    "sigmoid": r"C:\Users\cm120\Project\VLM_SNN_Research\MBE\mbe_models\activation\mbe_sigmoid_model.pth",
    "tanh": r"C:\Users\cm120\Project\VLM_SNN_Research\MBE\mbe_models\activation\mbe_tanh_model.pth",
}


@dataclass(frozen=True)
class ExternalMBEActivationConfig:
    target_name: str = "gelu"
    repo_path: str = r"C:\Users\cm120\Project\VLM_SNN_Research\MBE"
    checkpoint_path: str | None = None
    device: str = "cpu"
    chunk_elements: int = 8192


def _checkpoint_for(config: ExternalMBEActivationConfig) -> Path:
    if config.checkpoint_path is not None:
        return Path(config.checkpoint_path)
    target = config.target_name.lower()
    if target not in _DEFAULT_CHECKPOINTS:
        raise ValueError(f"No default MBE activation checkpoint for target: {config.target_name}")
    return Path(_DEFAULT_CHECKPOINTS[target])


def external_mbe_activation(
    x: np.ndarray,
    config: ExternalMBEActivationConfig | None = None,
) -> dict[str, np.ndarray | dict[str, float] | str]:
    """Run a trained external MBE activation checkpoint."""

    if config is None:
        config = ExternalMBEActivationConfig()

    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for external_mbe_activation.") from exc

    repo_path = Path(config.repo_path)
    if not repo_path.exists():
        raise FileNotFoundError(f"External MBE repo not found: {repo_path}")
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    from MBE_neurons import MBENeuron

    checkpoint_path = _checkpoint_for(config)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"MBE activation checkpoint not found: {checkpoint_path}")

    model = MBENeuron.load(str(checkpoint_path), device=config.device)
    model.eval()

    x32 = np.asarray(x, dtype=np.float32)
    original_shape = x32.shape
    x_flat = x32.reshape(-1)
    out_flat = np.empty_like(x_flat, dtype=np.float64)

    total_spike_events = 0.0
    total_compares = 0.0
    timesteps = int(getattr(model, "timesteps"))
    num_basis = int(getattr(model, "num_basis"))

    with torch.no_grad():
        for start in range(0, x_flat.size, config.chunk_elements):
            end = min(start + config.chunk_elements, x_flat.size)
            xt = torch.from_numpy(x_flat[start:end]).to(config.device).view(-1, 1)
            y, spikes, _ = model(xt, return_sequences=True)
            out_flat[start:end] = y.detach().cpu().numpy().reshape(-1)
            total_spike_events += float(spikes.sum().item())
            total_compares += float((end - start) * timesteps * num_basis)

    ops = {
        "spike_events": total_spike_events,
        "accumulate_ops": total_spike_events,
        "threshold_compares": total_compares,
        "lut_reads": 0.0,
    }
    return {
        "decoded": out_flat.reshape(original_shape),
        "ops": ops,
        "source": str(checkpoint_path),
        "timesteps": np.array(timesteps),
        "num_basis": np.array(num_basis),
    }


def error_summary(approx: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    approx = np.asarray(approx, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    abs_err = np.abs(approx - reference)
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(abs_err**2))),
        "max_abs": float(abs_err.max()),
    }
