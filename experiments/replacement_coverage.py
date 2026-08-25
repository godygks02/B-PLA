"""
Replacement coverage audit: what B-PLA actually converts, and what stays exact.

The submission plan requires every result to state its converted scope and its
exact remaining paths. Claiming "supported-operation conversion" without that
list overstates coverage, so this script enumerates it directly from the
converted model and weights each site by its multiply count.

No forward pass is needed: multiply counts follow from module shapes and the
sequence length, so this is cheap enough to run alongside the model probes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers.pytorch_utils import Conv1D

from modules.torch_bpla import (
    SharedBPLATables,
    TorchBPLAActivation,
    TorchBPLAConfig,
    TorchBPLAConv1D,
    TorchBPLAConv2d,
    TorchBPLALayerNorm,
    TorchBPLALinear,
    replace_attention_matmuls,
    replace_gpt2_conv1d_and_gelu,
    replace_layer_norms,
    replace_linear_and_gelu,
)


def _multiply_count(module: nn.Module, tokens: int) -> int:
    """Multiplies one forward pass of this module performs, per sequence."""

    if isinstance(module, (nn.Linear, TorchBPLALinear)):
        weight = module.weight
        return tokens * weight.shape[0] * weight.shape[1]
    if isinstance(module, (Conv1D, TorchBPLAConv1D)):
        return tokens * module.weight.shape[0] * module.weight.shape[1]
    if isinstance(module, (nn.Conv2d, TorchBPLAConv2d)):
        # ViT's patch embedding. One output position per non-class token.
        in_channels = (
            module.in_channels
            if isinstance(module, nn.Conv2d)
            else module.weight.shape[1]
        )
        per_position = module.out_channels * in_channels
        for size in module.kernel_size:
            per_position *= size
        return max(1, tokens - 1) * per_position
    return 0


def audit(model: nn.Module, tokens: int) -> dict[str, object]:
    converted: list[dict[str, object]] = []
    exact: list[dict[str, object]] = []

    for name, child in model.named_modules():
        count = _multiply_count(child, tokens)
        if count == 0:
            continue
        entry = {"name": name, "type": type(child).__name__, "multiplies": count}
        if isinstance(child, (TorchBPLALinear, TorchBPLAConv1D, TorchBPLAConv2d)):
            converted.append(entry)
        else:
            exact.append(entry)

    converted_total = sum(int(e["multiplies"]) for e in converted)
    exact_total = sum(int(e["multiplies"]) for e in exact)
    total = converted_total + exact_total

    # Group the exact remainder so the paper can name the paths, not just count.
    remaining: dict[str, dict[str, object]] = {}
    for entry in exact:
        # Collapse per-layer names: transformer.h.7.mlp.c_fc -> transformer.h.*.mlp.c_fc
        parts = [p if not p.isdigit() else "*" for p in str(entry["name"]).split(".")]
        key = ".".join(parts)
        bucket = remaining.setdefault(key, {"sites": 0, "multiplies": 0, "type": entry["type"]})
        bucket["sites"] = int(bucket["sites"]) + 1
        bucket["multiplies"] = int(bucket["multiplies"]) + int(entry["multiplies"])

    return {
        "tokens_per_sequence": tokens,
        "converted_sites": len(converted),
        "exact_sites": len(exact),
        "converted_multiplies": converted_total,
        "exact_multiplies": exact_total,
        "converted_fraction": converted_total / total if total else 0.0,
        "exact_remaining_paths": sorted(
            (
                {"path": key, **value}
                for key, value in remaining.items()
            ),
            key=lambda row: -int(row["multiplies"]),
        ),
        "bpla_activation_sites": sum(
            1 for m in model.modules() if isinstance(m, TorchBPLAActivation)
        ),
        "bpla_layernorm_sites": sum(
            1 for m in model.modules() if isinstance(m, TorchBPLALayerNorm)
        ),
        "attention_blocks_converted": int(getattr(model, "_bpla_attention_mode", "exact") != "exact"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="B-PLA replacement coverage audit.")
    parser.add_argument("--models", nargs="+", default=["vit", "gpt2"], choices=["vit", "gpt2"])
    parser.add_argument("--vit-model-id", default="google/vit-base-patch16-224")
    parser.add_argument("--gpt2-model-id", default="gpt2")
    parser.add_argument("--vit-tokens", type=int, default=197)
    parser.add_argument("--gpt2-tokens", type=int, default=256)
    parser.add_argument("--prefix-bits", type=int, default=4)
    parser.add_argument("--dyadic-terms", type=int, default=2)
    parser.add_argument("--scopes", nargs="+", default=["multiplication", "combined"])
    parser.add_argument("--replace-layernorm", action="store_true")
    parser.add_argument("--replace-lm-head", action="store_true")
    parser.add_argument("--replace-conv2d", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "replacement_coverage.json"
    )
    args = parser.parse_args()

    record: dict[str, object] = {"configuration": vars(args) | {"output": str(args.output)}, "audits": []}
    audits: list[dict[str, object]] = record["audits"]  # type: ignore[assignment]

    for model_name in args.models:
        for scope in args.scopes:
            config = TorchBPLAConfig(
                prefix_bits=args.prefix_bits, affine_path="dyadic", dyadic_terms=args.dyadic_terms
            )
            tables = SharedBPLATables(config)
            replace_multiplication = scope in {"multiplication", "combined"}
            replace_nonlinear = scope in {"nonlinear", "combined"}

            if model_name == "gpt2":
                from transformers import GPT2LMHeadModel

                model = GPT2LMHeadModel.from_pretrained(args.gpt2_model_id).eval()
                tokens = args.gpt2_tokens
                replace_gpt2_conv1d_and_gelu(
                    model,
                    config,
                    replace_multiplication,
                    replace_nonlinear,
                    None,
                    tables,
                    replace_lm_head=replace_multiplication and args.replace_lm_head,
                )
            else:
                from transformers import ViTForImageClassification

                model = ViTForImageClassification.from_pretrained(args.vit_model_id).eval()
                tokens = args.vit_tokens
                replace_linear_and_gelu(
                    model,
                    config,
                    replace_multiplication,
                    replace_nonlinear,
                    None,
                    tables,
                    replace_conv2d=replace_multiplication and args.replace_conv2d,
                )
                if replace_nonlinear:
                    for child in model.modules():
                        activation = getattr(child, "intermediate_act_fn", None)
                        if activation is not None and not isinstance(activation, TorchBPLAActivation):
                            child.intermediate_act_fn = TorchBPLAActivation("gelu", config, tables)

            if replace_multiplication:
                replace_attention_matmuls(
                    model, config, tables, mode="bpla-full", approximate_softmax=replace_nonlinear
                )
            if replace_nonlinear and args.replace_layernorm:
                replace_layer_norms(model, config, tables)

            result = audit(model, tokens) | {"model": model_name, "scope": scope}
            audits.append(result)
            del model

    args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")

    for result in audits:
        print(f"\n=== {result['model']} / scope={result['scope']} "
              f"({result['tokens_per_sequence']} tokens/sequence) ===")
        print(f"converted matmul sites : {result['converted_sites']}")
        print(f"exact matmul sites     : {result['exact_sites']}")
        print(f"converted multiplies   : {result['converted_multiplies']:,} "
              f"({100 * float(result['converted_fraction']):.2f}% of module multiplies)")
        print(f"B-PLA activation sites : {result['bpla_activation_sites']}")
        print(f"B-PLA LayerNorm sites  : {result['bpla_layernorm_sites']}")
        remaining = result["exact_remaining_paths"]
        assert isinstance(remaining, list)
        if remaining:
            print("exact remaining paths (largest first):")
            for row in remaining:
                print(f"  {row['path']:<44} {row['type']:<10} "
                      f"{row['sites']:>3} sites  {row['multiplies']:>14,} mults")
        else:
            print("exact remaining paths  : none among weighted modules")
        print("note: attention QK/PV multiplies are not counted here; they are "
              "activation-activation products with no weight module.")

    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
