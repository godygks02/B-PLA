"""
Compare an MBE-style multiplication baseline with B-PLA-SNN multiplication.

The MBE baseline follows the multiplication structure described in the MBE
paper summary, but uses deterministic identity encoding because trained MBE
parameters are not available in the local notes.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules import bpla_SNN_multiplier
from modules import bpla_multiplier
from modules import mbe_multiplier


ENERGY_PJ = {
    "fp32_mul": 3.70,
    "fp32_add": 0.90,
    "lut_read": 0.04,
    "spike_accumulate": 0.10,
    "threshold_compare": 0.02,
}


def estimate_energy_proxy(ops: dict[str, float]) -> float:
    return (
        ops.get("lut_reads", 0.0) * ENERGY_PJ["lut_read"]
        + ops.get("spike_events", 0.0) * ENERGY_PJ["spike_accumulate"]
        + ops.get("pairwise_spike_interactions", 0.0) * ENERGY_PJ["spike_accumulate"]
        + ops.get("accumulate_ops", 0.0) * ENERGY_PJ["spike_accumulate"]
        + ops.get("threshold_compares", 0.0) * ENERGY_PJ["threshold_compare"]
    )


def _attention_scores_with_multiplier(
    q: np.ndarray,
    k: np.ndarray,
    method: str,
    bpla_cfg: bpla_SNN_multiplier.BPLASpikingMultiplierConfig,
    mbe_cfg: mbe_multiplier.MBEIdentityConfig,
    external_mbe_cfg: mbe_multiplier.ExternalMBEConfig | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    batch, heads, seq, dim = q.shape
    q_exp = q[:, :, :, None, :]
    k_exp = k[:, :, None, :, :]
    q_broad, k_broad = np.broadcast_arrays(q_exp, k_exp)
    if method == "bpla":
        result = bpla_SNN_multiplier.bpla_snn_multiply(q_broad, k_broad, bpla_cfg)
    elif method == "external_mbe":
        if external_mbe_cfg is None:
            raise ValueError("external_mbe_cfg is required for external_mbe.")
        result = mbe_multiplier.external_mbe_multiply(q_broad, k_broad, external_mbe_cfg)
    elif method == "mbe":
        result = mbe_multiplier.mbe_multiply(q_broad, k_broad, mbe_cfg)
    else:
        raise ValueError("method must be 'bpla' or 'mbe'.")
    products = np.asarray(result["decoded"], dtype=np.float64)
    scores = np.sum(products, axis=-1) / np.sqrt(float(dim))
    return scores, result["ops"]


def fp_multiplication_sweep(
    samples: int = 8192,
    seed: int = 11,
    use_external_mbe: bool = False,
) -> tuple[list[dict[str, float | str]], str | None]:
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, size=samples).astype(np.float32)
    b = rng.normal(0.0, 1.0, size=samples).astype(np.float32)
    ref = bpla_multiplier.exact_multiply(a, b)
    rows: list[dict[str, float | str]] = []
    external_note = None

    if use_external_mbe:
        try:
            external_cfg = mbe_multiplier.ExternalMBEConfig()
            result = mbe_multiplier.external_mbe_multiply(a, b, external_cfg)
            metrics = mbe_multiplier.error_summary(result["decoded"], ref)
            ops = result["ops"]
            rows.append(
                {
                    "method": "external_mbe",
                    "encoding": "trained_mbe_identity",
                    "timesteps": float(np.asarray(result["timesteps"])),
                    "num_basis": float(np.asarray(result["num_basis"])),
                    "mae": metrics["mae"],
                    "mean_rel": metrics["mean_rel"],
                    "p99_rel": metrics["p99_rel"],
                    "spike_events": ops["spike_events"],
                    "pairwise_spike_interactions": ops["pairwise_spike_interactions"],
                    "energy_proxy_pj": estimate_energy_proxy(ops),
                    "source": str(result["source"]),
                }
            )
        except Exception as exc:
            external_note = f"External MBE unavailable; used NumPy structural baselines only. Reason: {exc}"

    for timesteps in [4, 8, 12, 16]:
        for encoding_name, exponential_decay in [("exponential", True), ("binary", False)]:
            mbe_cfg = mbe_multiplier.MBEIdentityConfig(
                timesteps=timesteps,
                value_range=4.0,
                exponential_decay=exponential_decay,
            )
            result = mbe_multiplier.mbe_multiply(a, b, mbe_cfg)
            metrics = mbe_multiplier.error_summary(result["decoded"], ref)
            ops = result["ops"]
            rows.append(
                {
                    "method": "mbe_style",
                    "encoding": encoding_name,
                    "timesteps": float(timesteps),
                    "mae": metrics["mae"],
                    "mean_rel": metrics["mean_rel"],
                    "p99_rel": metrics["p99_rel"],
                    "spike_events": ops["spike_events"],
                    "pairwise_spike_interactions": ops["pairwise_spike_interactions"],
                    "energy_proxy_pj": estimate_energy_proxy(ops),
                }
            )

    for prefix_bits in [2, 3, 4, 5]:
        bpla_cfg = bpla_SNN_multiplier.BPLASpikingMultiplierConfig(
            mantissa_bits=16,
            prefix_bits=prefix_bits,
            threshold=1.0 / 65536.0,
            neuron_type="fs",
            progressive_levels=True,
        )
        result = bpla_SNN_multiplier.bpla_snn_multiply(a, b, bpla_cfg)
        metrics = bpla_SNN_multiplier.error_summary(result["decoded"], ref)
        ops = result["ops"]
        rows.append(
            {
                "method": "bpla_snn",
                "encoding": "mantissa_prefix",
                "prefix_bits": float(prefix_bits),
                "timesteps": float(bpla_cfg.mantissa_bits),
                "mae": metrics["mae"],
                "mean_rel": metrics["mean_rel"],
                "p99_rel": metrics["p99_rel"],
                "spike_events": ops["spike_events"],
                "pairwise_spike_interactions": 0.0,
                "energy_proxy_pj": estimate_energy_proxy(ops),
            }
        )
    return rows, external_note


def attention_score_sweep(
    batch: int = 2,
    heads: int = 2,
    seq: int = 16,
    dim: int = 32,
    seed: int = 23,
    use_external_mbe: bool = False,
) -> tuple[list[dict[str, float | str]], str | None]:
    rng = np.random.default_rng(seed)
    q = rng.normal(0.0, 0.5, size=(batch, heads, seq, dim)).astype(np.float32)
    k = rng.normal(0.0, 0.5, size=(batch, heads, seq, dim)).astype(np.float32)
    ref = (q.astype(np.float64) @ np.swapaxes(k.astype(np.float64), -1, -2)) / np.sqrt(float(dim))

    rows: list[dict[str, float | str]] = []
    external_note = None
    mbe_cfgs = [
        ("mbe_style_exponential", mbe_multiplier.MBEIdentityConfig(timesteps=12, value_range=4.0, exponential_decay=True)),
        ("mbe_style_binary", mbe_multiplier.MBEIdentityConfig(timesteps=12, value_range=4.0, exponential_decay=False)),
    ]
    bpla_cfg = bpla_SNN_multiplier.BPLASpikingMultiplierConfig(
        mantissa_bits=16,
        prefix_bits=4,
        threshold=1.0 / 65536.0,
        neuron_type="fs",
        progressive_levels=True,
    )

    method_specs = [
        *[(name, "mbe", cfg) for name, cfg in mbe_cfgs],
        ("bpla_snn", "bpla", mbe_cfgs[0][1]),
    ]
    if use_external_mbe:
        method_specs.insert(0, ("external_mbe", "external_mbe", mbe_cfgs[0][1]))

    for method_name, method, mbe_cfg in method_specs:
        try:
            scores, ops = _attention_scores_with_multiplier(
                q,
                k,
                method,
                bpla_cfg,
                mbe_cfg,
                mbe_multiplier.ExternalMBEConfig() if method == "external_mbe" else None,
            )
        except Exception as exc:
            if method == "external_mbe":
                external_note = f"External MBE attention run unavailable. Reason: {exc}"
                continue
            raise
        err = scores - ref
        abs_err = np.abs(err)
        rows.append(
            {
                "method": method_name,
                "batch": float(batch),
                "heads": float(heads),
                "seq": float(seq),
                "dim": float(dim),
                "score_mae": float(abs_err.mean()),
                "score_rmse": float(np.sqrt(np.mean(err**2))),
                "score_max_abs": float(abs_err.max()),
                "spike_events": ops["spike_events"],
                "pairwise_spike_interactions": ops.get("pairwise_spike_interactions", 0.0),
                "energy_proxy_pj": estimate_energy_proxy(ops),
            }
        )
    return rows, external_note


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_external_mbe",
        action="store_true",
        help="Use C:/Users/cm120/Project/VLM_SNN_Research/MBE when PyTorch is available.",
    )
    args = parser.parse_args()
    fp_rows, fp_external_note = fp_multiplication_sweep(use_external_mbe=args.use_external_mbe)
    attention_rows, attention_external_note = attention_score_sweep(use_external_mbe=args.use_external_mbe)
    results = {
        "fp_multiplication": fp_rows,
        "attention_scores": attention_rows,
        "energy_assumptions_pj": ENERGY_PJ,
        "mbe_baseline_note": (
            "mbe_style rows use deterministic NumPy identity encoders. "
            "external_mbe rows, when present, use the user's PyTorch MBE implementation."
        ),
        "external_mbe_notes": [note for note in [fp_external_note, attention_external_note] if note],
    }
    print(json.dumps(results, indent=2, sort_keys=True))
    out_path = Path(__file__).resolve().parent / "mbe_vs_bpla_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
