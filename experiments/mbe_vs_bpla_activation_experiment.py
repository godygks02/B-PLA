"""
Compare trained MBE activation checkpoints with B-PLA-SNN activation.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules import bpla_activation
from modules import bpla_SNN_activation
from modules import mbe_activation


ENERGY_PJ = {
    "fp32_mul": 3.70,
    "fp32_add": 0.90,
    "lut_read": 0.04,
    "spike_accumulate": 0.10,
    "threshold_compare": 0.02,
}


DEFAULT_RANGES = {
    "gelu": (-4.0, 4.0),
    "tanh": (-4.0, 4.0),
    "sigmoid": (-6.0, 6.0),
}


def estimate_energy_proxy(ops: dict[str, float]) -> float:
    return (
        ops.get("lut_reads", 0.0) * ENERGY_PJ["lut_read"]
        + ops.get("spike_events", 0.0) * ENERGY_PJ["spike_accumulate"]
        + ops.get("accumulate_ops", 0.0) * ENERGY_PJ["spike_accumulate"]
        + ops.get("threshold_compares", 0.0) * ENERGY_PJ["threshold_compare"]
    )


def _plot_curves(
    target: str,
    x: np.ndarray,
    exact: np.ndarray,
    mbe: np.ndarray | None,
    bpla: np.ndarray,
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(x, exact, label="Exact", linewidth=2.0, color="black")
    if mbe is not None:
        plt.plot(x, mbe, label="MBE checkpoint", linestyle="--", linewidth=1.8)
    plt.plot(x, bpla, label="B-PLA-SNN", linestyle="-.", linewidth=1.8)
    plt.title(f"{target.upper()} approximation")
    plt.xlabel("x")
    plt.ylabel(f"{target}(x)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def run_target(
    target: str,
    samples: int,
    prefix_bits: int,
    bit_width: int,
    use_external_mbe: bool,
) -> tuple[dict[str, float | str], dict[str, np.ndarray]]:
    x_min, x_max = DEFAULT_RANGES[target]
    x = np.linspace(x_min, x_max, samples, dtype=np.float64)
    exact = bpla_activation.exact_activation(x, target)

    bpla_cfg = bpla_SNN_activation.BPLASpikingNeuronConfig(
        neuron_type="fs",
        target_name=target,
        threshold=1.0 / float(1 << max(1, bit_width - 4)),
        bit_width=bit_width,
        fractional_bits=bit_width - 4,
        prefix_bits=prefix_bits,
        progressive_levels=True,
        x_min=x_min,
        x_max=x_max,
    )
    bpla_result = bpla_SNN_activation.bpla_snn_activation(x, bpla_cfg)
    bpla_decoded = np.asarray(bpla_result["decoded"], dtype=np.float64)
    bpla_metrics = bpla_SNN_activation.error_summary(bpla_decoded, exact)

    row: dict[str, float | str] = {
        "target": target,
        "samples": float(samples),
        "x_min": x_min,
        "x_max": x_max,
        "bpla_prefix_bits": float(prefix_bits),
        "bpla_timesteps": float(bit_width),
        "bpla_mae": bpla_metrics["mae"],
        "bpla_rmse": bpla_metrics["rmse"],
        "bpla_max_abs": bpla_metrics["max_abs"],
        "bpla_spike_events": float(bpla_result["ops"]["spike_events"]),
        "bpla_energy_proxy_pj": estimate_energy_proxy(bpla_result["ops"]),
    }
    curves = {"x": x, "exact": exact, "bpla": bpla_decoded}

    if use_external_mbe:
        try:
            mbe_cfg = mbe_activation.ExternalMBEActivationConfig(target_name=target)
            mbe_result = mbe_activation.external_mbe_activation(x, mbe_cfg)
            mbe_decoded = np.asarray(mbe_result["decoded"], dtype=np.float64)
            mbe_metrics = mbe_activation.error_summary(mbe_decoded, exact)
            row.update(
                {
                    "mbe_available": "yes",
                    "mbe_source": str(mbe_result["source"]),
                    "mbe_timesteps": float(np.asarray(mbe_result["timesteps"])),
                    "mbe_num_basis": float(np.asarray(mbe_result["num_basis"])),
                    "mbe_mae": mbe_metrics["mae"],
                    "mbe_rmse": mbe_metrics["rmse"],
                    "mbe_max_abs": mbe_metrics["max_abs"],
                    "mbe_spike_events": float(mbe_result["ops"]["spike_events"]),
                    "mbe_energy_proxy_pj": estimate_energy_proxy(mbe_result["ops"]),
                }
            )
            curves["mbe"] = mbe_decoded
        except Exception as exc:
            row.update({"mbe_available": "no", "mbe_error": str(exc)})
    else:
        row.update({"mbe_available": "not_requested"})

    return row, curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=["gelu", "tanh", "sigmoid"])
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--prefix_bits", type=int, default=4)
    parser.add_argument("--bit_width", type=int, default=16)
    parser.add_argument("--use_external_mbe", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    rows = []
    curves_by_target = {}
    for target in args.targets:
        row, curves = run_target(
            target=target,
            samples=args.samples,
            prefix_bits=args.prefix_bits,
            bit_width=args.bit_width,
            use_external_mbe=args.use_external_mbe,
        )
        rows.append(row)
        curves_by_target[target] = curves

    out_dir = Path(__file__).resolve().parent
    json_path = out_dir / "mbe_vs_bpla_activation_results.json"
    csv_path = out_dir / "mbe_vs_bpla_activation_results.csv"
    json_path.write_text(
        json.dumps(
            {
                "rows": rows,
                "energy_assumptions_pj": ENERGY_PJ,
                "note": "MBE rows use external trained checkpoints when --use_external_mbe is set.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.plot:
        plot_dir = out_dir / "plots" / "activation_compare"
        for target, curves in curves_by_target.items():
            _plot_curves(
                target,
                curves["x"],
                curves["exact"],
                curves.get("mbe"),
                curves["bpla"],
                plot_dir / f"{target}_mbe_vs_bpla.png",
            )

    print(json.dumps(rows, indent=2, sort_keys=True))
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
