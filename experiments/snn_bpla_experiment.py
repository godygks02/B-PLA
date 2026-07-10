"""
Preliminary B-PLA-SNN experiments.

This script evaluates the prefix-routed PLA spiking neuron at operator level
and on a small NumPy-only toy MLP. Reported energy is an operation-level proxy,
not measured hardware power.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules import bpla_activation
from modules import bpla_SNN_activation
from modules import bpla_SNN_multiplier
from modules import bpla_multiplier


ENERGY_PJ = {
    "fp32_mul": 3.70,
    "fp32_add": 0.90,
    "lut_read": 0.04,
    "spike_accumulate": 0.10,
    "threshold_compare": 0.02,
}


def _merge_ops(*ops_dicts: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for ops in ops_dicts:
        for key, value in ops.items():
            merged[key] = merged.get(key, 0.0) + float(value)
    return merged


def estimate_snn_energy_proxy(ops: dict[str, float]) -> float:
    return (
        ops.get("lut_reads", 0.0) * ENERGY_PJ["lut_read"]
        + ops.get("spike_events", 0.0) * ENERGY_PJ["spike_accumulate"]
        + ops.get("accumulate_ops", 0.0) * ENERGY_PJ["spike_accumulate"]
        + ops.get("threshold_compares", 0.0) * ENERGY_PJ["threshold_compare"]
    )


def activation_sweep() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    x = np.linspace(-4.0, 4.0, 4096)
    for target in ["relu", "gelu", "quick_gelu"]:
        ref = bpla_activation.exact_activation(x, target)
        for neuron_type in ["fs", "if"]:
            progressive_modes = [True, False] if neuron_type == "fs" else [False]
            for progressive in progressive_modes:
                cfg = bpla_SNN_activation.BPLASpikingNeuronConfig(
                    neuron_type=neuron_type,
                    target_name=target,
                    threshold=1.0 / 256.0,
                    progressive_levels=progressive,
                )
                result = bpla_SNN_activation.bpla_snn_activation(x, cfg)
                metrics = bpla_SNN_activation.error_summary(result["decoded"], ref)
                ops = result["ops"]
                fp_energy = x.size * (ENERGY_PJ["fp32_mul"] + ENERGY_PJ["fp32_add"])
                snn_energy = estimate_snn_energy_proxy(ops)
                rows.append(
                    {
                        "target": target,
                        "neuron": neuron_type,
                        "progressive": progressive,
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "max_abs": metrics["max_abs"],
                        "spike_events": ops["spike_events"],
                        "snn_energy_pj": snn_energy,
                        "fp_energy_pj": fp_energy,
                        "energy_ratio": snn_energy / fp_energy,
                    }
                )
    return rows


def multiplier_sweep(samples: int = 4096, seed: int = 21) -> list[dict[str, float | str]]:
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, size=samples).astype(np.float32)
    b = rng.normal(0.0, 1.0, size=samples).astype(np.float32)
    ref = bpla_multiplier.exact_multiply(a, b)
    rows: list[dict[str, float | str]] = []
    for neuron_type in ["fs", "if"]:
        progressive_modes = [True, False] if neuron_type == "fs" else [False]
        for progressive in progressive_modes:
            cfg = bpla_SNN_multiplier.BPLASpikingMultiplierConfig(
                neuron_type=neuron_type,
                mantissa_bits=12,
                prefix_bits=4,
                threshold=1.0 / 4096.0,
                progressive_levels=progressive,
            )
            result = bpla_SNN_multiplier.bpla_snn_multiply(a, b, cfg)
            metrics = bpla_SNN_multiplier.error_summary(result["decoded"], ref)
            ops = result["ops"]
            fp_energy = samples * ENERGY_PJ["fp32_mul"]
            snn_energy = estimate_snn_energy_proxy(ops)
            rows.append(
                {
                    "neuron": neuron_type,
                    "progressive": progressive,
                    "mean_rel": metrics["mean_rel"],
                    "p99_rel": metrics["p99_rel"],
                    "max_rel": metrics["max_rel"],
                    "spike_events": ops["spike_events"],
                    "snn_energy_pj": snn_energy,
                    "fp_energy_pj": fp_energy,
                    "energy_ratio": snn_energy / fp_energy,
                }
            )
    return rows


def progressive_level_sweep(samples: int = 4096, seed: int = 31) -> dict[str, list[dict[str, float]]]:
    x = np.linspace(-4.0, 4.0, samples)
    ref_act = bpla_activation.exact_activation(x, "gelu")
    act_cfg = bpla_SNN_activation.BPLASpikingNeuronConfig(
        neuron_type="fs",
        target_name="gelu",
        threshold=1.0 / 256.0,
        prefix_bits=4,
        progressive_levels=True,
    )
    act_result = bpla_SNN_activation.bpla_snn_activation(x, act_cfg)
    activation_rows: list[dict[str, float]] = []
    for level in range(act_result["level_values"].shape[-1]):
        decoded, _, _, events = bpla_SNN_activation._emit_few_spikes(
            act_result["level_values"][..., level],
            act_cfg.threshold,
            act_cfg.bit_width,
        )
        metrics = bpla_SNN_activation.error_summary(decoded, ref_act)
        activation_rows.append(
            {
                "level": float(level + 1),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "spike_events": float(np.sum(events)),
            }
        )

    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, size=samples).astype(np.float32)
    b = rng.normal(0.0, 1.0, size=samples).astype(np.float32)
    ref_mul = bpla_multiplier.exact_multiply(a, b)
    mul_cfg = bpla_SNN_multiplier.BPLASpikingMultiplierConfig(
        neuron_type="fs",
        mantissa_bits=12,
        prefix_bits=4,
        threshold=1.0 / 4096.0,
        progressive_levels=True,
    )
    mul_result = bpla_SNN_multiplier.bpla_snn_multiply(a, b, mul_cfg)
    pa = bpla_multiplier.decompose_float32(a)
    pb = bpla_multiplier.decompose_float32(b)
    enc_a = bpla_SNN_multiplier.encode_mantissa_bitplanes(pa.fraction_q23, mul_cfg.mantissa_bits)
    enc_b = bpla_SNN_multiplier.encode_mantissa_bitplanes(pb.fraction_q23, mul_cfg.mantissa_bits)
    multiplier_rows: list[dict[str, float]] = []
    for level in range(mul_result["level_values"].shape[-1]):
        cross, events = bpla_SNN_multiplier._emit_few_spikes(
            mul_result["level_values"][..., level],
            mul_cfg.threshold,
            mul_cfg.mantissa_bits,
        )
        sign = pa.sign ^ pb.sign
        mantissa = 1.0 + enc_a["decoded"] + enc_b["decoded"] + cross
        decoded = bpla_multiplier._assemble_product(
            sign,
            pa.exponent + pb.exponent,
            mantissa,
            pa.normal & pb.normal,
            ref_mul,
        )
        metrics = bpla_SNN_multiplier.error_summary(decoded, ref_mul)
        multiplier_rows.append(
            {
                "level": float(level + 1),
                "mean_rel": metrics["mean_rel"],
                "p99_rel": metrics["p99_rel"],
                "spike_events": float(np.sum(events)),
            }
        )
    return {"activation_gelu": activation_rows, "multiplier": multiplier_rows}


def toy_mlp_conversion(samples: int = 512, seed: int = 5) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(samples, 32))
    w1 = rng.normal(0.0, 0.25, size=(32, 48))
    b1 = rng.normal(0.0, 0.05, size=(48,))
    w2 = rng.normal(0.0, 0.25, size=(48, 10))
    b2 = rng.normal(0.0, 0.05, size=(10,))

    h = x @ w1 + b1
    h = np.clip(h, -4.0, 4.0)
    h_ref = bpla_activation.exact_activation(h, "gelu")
    logits_ref = h_ref @ w2 + b2

    cfg = bpla_SNN_activation.BPLASpikingNeuronConfig(
        neuron_type="fs",
        target_name="gelu",
        threshold=1.0 / 256.0,
        progressive_levels=True,
    )
    snn_result = bpla_SNN_activation.bpla_snn_activation(h, cfg)
    logits_snn = snn_result["decoded"] @ w2 + b2

    ref_pred = np.argmax(logits_ref, axis=1)
    snn_pred = np.argmax(logits_snn, axis=1)
    agreement = float(np.mean(ref_pred == snn_pred))
    logit_rmse = float(np.sqrt(np.mean((logits_snn - logits_ref) ** 2)))

    fp_ops = float(samples * (32 * 48 + 48 * 10))
    fp_energy = fp_ops * (ENERGY_PJ["fp32_mul"] + ENERGY_PJ["fp32_add"])
    snn_activation_energy = estimate_snn_energy_proxy(snn_result["ops"])
    return {
        "samples": float(samples),
        "top1_agreement": agreement,
        "logit_rmse": logit_rmse,
        "activation_spike_events": float(snn_result["ops"]["spike_events"]),
        "snn_activation_energy_pj": snn_activation_energy,
        "fp_mlp_energy_proxy_pj": fp_energy,
        "activation_energy_over_mlp_fp_proxy": snn_activation_energy / fp_energy,
    }


def print_table(title: str, rows: list[dict[str, float | str]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for row in rows:
        print(json.dumps(row, indent=None, sort_keys=True))


def main() -> None:
    results = {
        "activation": activation_sweep(),
        "multiplier": multiplier_sweep(),
        "progressive_levels": progressive_level_sweep(),
        "toy_mlp": toy_mlp_conversion(),
        "energy_assumptions_pj": ENERGY_PJ,
    }
    print_table("Activation sweep", results["activation"])
    print_table("Multiplier sweep", results["multiplier"])
    print_table("Progressive GELU levels", results["progressive_levels"]["activation_gelu"])
    print_table("Progressive multiplier levels", results["progressive_levels"]["multiplier"])
    print("\nToy MLP conversion")
    print("------------------")
    print(json.dumps(results["toy_mlp"], indent=None, sort_keys=True))

    out_path = Path(__file__).resolve().parent / "snn_bpla_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
