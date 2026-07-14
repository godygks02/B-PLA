"""Standalone compute-only energy experiment for B-PLA primitives and models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from modules.compute_energy import (
    BPLAComputeConfig,
    ComputeEnergyTablePJ,
    ComputeWorkload,
    bpla_gelu_energy_pj,
    bpla_multiplier_energy_pj,
    estimate_workload_compute_energy,
    format_compute_energy_report,
    fp32_gelu_energy_pj,
    mlp_workload,
)


def vit_base_workload(attention_mode: str, replace_linear: bool, replace_gelu: bool) -> ComputeWorkload:
    layers, tokens, hidden, intermediate = 12, 197, 768, 3072
    projection = layers * 4 * tokens * hidden * hidden
    qk = layers * tokens * tokens * hidden
    pv = qk
    mlp = layers * 2 * tokens * hidden * intermediate
    classifier = hidden * 1000
    patch = 196 * 3 * 16 * 16 * hidden
    selected_attention = (qk if attention_mode in {"bpla-qk", "bpla-full"} else 0) + (
        pv if attention_mode in {"bpla-pv", "bpla-full"} else 0
    )
    linear_sites = projection + mlp + classifier
    gelu = layers * tokens * intermediate
    return ComputeWorkload(
        multiply_sites=linear_sites + qk + pv + patch,
        bpla_multiply_sites=(linear_sites if replace_linear else 0) + selected_attention,
        gelu_sites=gelu,
        bpla_gelu_sites=gelu if replace_gelu else 0,
        label="ViT-Base/16-224 / image",
    )


def gpt2_small_workload(
    sequence_length: int,
    attention_mode: str,
    replace_conv1d: bool,
    replace_gelu: bool,
) -> ComputeWorkload:
    layers, hidden, intermediate, vocab = 12, 768, 3072, 50_257
    conv1d = layers * (
        4 * sequence_length * hidden * hidden
        + 2 * sequence_length * hidden * intermediate
    )
    qk = layers * sequence_length * sequence_length * hidden
    pv = qk
    lm_head = sequence_length * hidden * vocab
    selected_attention = (qk if attention_mode in {"bpla-qk", "bpla-full"} else 0) + (
        pv if attention_mode in {"bpla-pv", "bpla-full"} else 0
    )
    gelu = layers * sequence_length * intermediate
    return ComputeWorkload(
        multiply_sites=conv1d + qk + pv + lm_head,
        bpla_multiply_sites=(conv1d if replace_conv1d else 0) + selected_attention,
        gelu_sites=gelu,
        bpla_gelu_sites=gelu if replace_gelu else 0,
        label=f"GPT-2 small prefill / sequence (length={sequence_length})",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute-only theoretical B-PLA energy comparison.")
    parser.add_argument("--affine-path", choices=["float", "dyadic"], default="dyadic")
    parser.add_argument("--dyadic-terms", type=int, default=2)
    parser.add_argument("--mantissa-bits", type=int, default=24)
    parser.add_argument("--shift-energy-pj", type=float, default=0.0)
    parser.add_argument("--control-energy-pj", type=float, default=0.005)
    parser.add_argument("--tanh-energy-pj", type=float, default=0.0)
    parser.add_argument("--gpt2-sequence-length", type=int, default=256)
    parser.add_argument("--attention-mode", choices=["exact", "bpla-qk", "bpla-pv", "bpla-full"], default="bpla-full")
    parser.add_argument("--mlp-input-dim", type=int, default=784)
    parser.add_argument("--mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--mlp-output-dim", type=int, default=10)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BPLAComputeConfig(args.affine_path, args.dyadic_terms, args.mantissa_bits)
    table = ComputeEnergyTablePJ(
        fixed_shift=args.shift_energy_pj,
        small_control=args.control_energy_pj,
        fp32_tanh=args.tanh_energy_pj,
    )
    primitive = {
        "fp32_gelu": fp32_gelu_energy_pj(table),
        "bpla_multiplier": bpla_multiplier_energy_pj(config, table),
        "bpla_gelu": bpla_gelu_energy_pj(config, table),
    }
    workloads = [
        mlp_workload(args.mlp_input_dim, args.mlp_hidden_dim, args.mlp_output_dim),
        vit_base_workload(args.attention_mode, replace_linear=True, replace_gelu=True),
        gpt2_small_workload(args.gpt2_sequence_length, args.attention_mode, True, True),
    ]
    reports = [estimate_workload_compute_energy(workload, config, table) for workload in workloads]

    multiplier = primitive["bpla_multiplier"]
    gelu = primitive["bpla_gelu"]
    print("\nB-PLA primitive compute-only energy")
    print("=" * 72)
    print(f"FP32 multiply            : {table.fp32_mul:.6f} pJ")
    print(f"B-PLA multiply           : {multiplier['total_pj']:.6f} pJ")
    print(f"B-PLA / FP32 multiply    : {multiplier['ratio_to_fp32_mul']:.6f}x")
    print(f"FP32 tanh-GELU lower bound: {primitive['fp32_gelu']['total_pj']:.6f} pJ")
    print(f"B-PLA GELU               : {gelu['total_pj']:.6f} pJ")
    print(f"B-PLA / FP32 GELU        : {gelu['ratio_to_fp32_gelu']:.6f}x")
    print("Memory/LUT/interconnect/leakage are excluded; tanh cost defaults to zero.")
    for report in reports:
        print("\n" + format_compute_energy_report(report))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "assumptions": {"energy_table_pj": table.__dict__, "bpla": config.__dict__},
            "primitive": primitive,
            "workloads": reports,
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
