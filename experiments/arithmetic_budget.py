"""
Where the shift-add budget actually goes: multiplication versus nonlinear.

The term-budget sweep found that the nonlinear tables need T=4 where the
multiplier is done at T=2, and argued that giving them their own budget is
nearly free because nonlinear operations are far rarer than multiplications.
"Far rarer" was an estimate. This counts it.

Counts come from the model's shapes and from the actual compositions in
``modules/torch_bpla.py`` -- the Softmax and LayerNorm figures follow
``bpla_softmax_torch`` and ``bpla_layer_norm_torch`` operation by operation
rather than from a textbook formula, so they track the implementation.

Per-invocation shift-add cost, for the separable multiplier and the
one-dimensional tables:

    multiplication          2 * T_mult   (nu*m1 and mu*(m2-nu))
    1-D nonlinear table         T_nl     (one slope * delta product)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
import sys


@dataclass
class Counts:
    """Invocations of each replaceable primitive, per input sequence.

    ``exact_remaining`` holds float multiplies that survive even at full
    conversion, so the report cannot quietly imply there are none. It is empty
    now that the Softmax and attention scalings are routed through the
    multiplier; what remains outside it is addition, which the method never
    claimed to replace.
    """

    multiplications: dict[str, int] = field(default_factory=dict)
    nonlinear: dict[str, int] = field(default_factory=dict)
    exact_remaining: dict[str, int] = field(default_factory=dict)

    @property
    def total_multiplications(self) -> int:
        return sum(self.multiplications.values())

    @property
    def total_nonlinear(self) -> int:
        return sum(self.nonlinear.values())


def gpt2_counts(layers: int, hidden: int, heads: int, ffn: int, seq: int, vocab: int) -> Counts:
    head_dim = hidden // heads
    counts = Counts()

    # Weighted projections. GPT-2's blocks are Conv1D; lm_head is a Linear.
    counts.multiplications["qkv projection"] = layers * seq * hidden * 3 * hidden
    counts.multiplications["attention output projection"] = layers * seq * hidden * hidden
    counts.multiplications["mlp"] = layers * seq * hidden * ffn * 2
    counts.multiplications["output projection (lm_head)"] = seq * hidden * vocab

    # Attention scores and value aggregation are activation-activation products.
    counts.multiplications["attention QK^T"] = layers * heads * seq * seq * head_dim
    counts.multiplications["attention PV"] = layers * heads * seq * seq * head_dim

    # bpla_softmax_torch: one multiply per element for the normalization, and
    # one more for the correction pass.
    softmax_rows = layers * heads * seq
    counts.multiplications["softmax normalization"] = 2 * softmax_rows * seq

    # bpla_layer_norm_torch: mean, squared, variance, normalized, weight.
    norm_rows = layers * 2 * seq + seq
    counts.multiplications["layernorm"] = norm_rows * (3 * hidden + 2)

    # One-dimensional table lookups.
    counts.nonlinear["GELU"] = layers * seq * ffn
    counts.nonlinear["exp2 (softmax)"] = layers * heads * seq * seq
    counts.nonlinear["reciprocal (softmax)"] = 2 * softmax_rows
    counts.nonlinear["rsqrt (layernorm)"] = norm_rows

    # Both of these go through the multiplier: the log2(e) scaling inside
    # Softmax, and the 1/sqrt(head_dim) attention scaling. The latter is a power
    # of two for these models and the multiplier is exact there, but it is
    # routed rather than assumed away because that is a property of head_dim.
    counts.multiplications["log2(e) scaling (softmax)"] = layers * heads * seq * seq
    counts.multiplications["attention scaling"] = layers * heads * seq * seq
    return counts


def vit_counts(layers: int, hidden: int, heads: int, ffn: int, tokens: int, classes: int,
               patch: int, channels: int) -> Counts:
    head_dim = hidden // heads
    counts = Counts()

    counts.multiplications["patch embedding"] = (tokens - 1) * hidden * channels * patch * patch
    counts.multiplications["qkv projection"] = layers * tokens * hidden * 3 * hidden
    counts.multiplications["attention output projection"] = layers * tokens * hidden * hidden
    counts.multiplications["mlp"] = layers * tokens * hidden * ffn * 2
    counts.multiplications["classifier"] = hidden * classes

    counts.multiplications["attention QK^T"] = layers * heads * tokens * tokens * head_dim
    counts.multiplications["attention PV"] = layers * heads * tokens * tokens * head_dim

    softmax_rows = layers * heads * tokens
    counts.multiplications["softmax normalization"] = 2 * softmax_rows * tokens

    norm_rows = layers * 2 * tokens + tokens
    counts.multiplications["layernorm"] = norm_rows * (3 * hidden + 2)

    counts.nonlinear["GELU"] = layers * tokens * ffn
    counts.nonlinear["exp2 (softmax)"] = layers * heads * tokens * tokens
    counts.nonlinear["reciprocal (softmax)"] = 2 * softmax_rows
    counts.nonlinear["rsqrt (layernorm)"] = norm_rows

    # Both of these go through the multiplier: the log2(e) scaling inside
    # Softmax, and the 1/sqrt(head_dim) attention scaling. The latter is a power
    # of two for these models and the multiplier is exact there, but it is
    # routed rather than assumed away because that is a property of head_dim.
    counts.multiplications["log2(e) scaling (softmax)"] = layers * heads * tokens * tokens
    counts.multiplications["attention scaling"] = layers * heads * tokens * tokens
    return counts


def report(name: str, counts: Counts, mult_terms: int, nonlinear_terms: int, baseline_terms: int) -> dict:
    mult_ops = counts.total_multiplications
    nl_ops = counts.total_nonlinear
    mult_cost = mult_ops * 2 * mult_terms
    nl_cost = nl_ops * nonlinear_terms
    nl_cost_baseline = nl_ops * baseline_terms
    total = mult_cost + nl_cost

    print(f"\n=== {name} ===")
    print(f"{'primitive':<34}{'invocations':>16}{'share':>9}")
    print("-" * 59)
    for label, value in sorted(counts.multiplications.items(), key=lambda kv: -kv[1]):
        print(f"  mult: {label:<27}{value:>16,}{100*value/mult_ops:>8.2f}%")
    print(f"  {'MULTIPLICATION TOTAL':<32}{mult_ops:>16,}")
    print()
    for label, value in sorted(counts.nonlinear.items(), key=lambda kv: -kv[1]):
        print(f"  nl:   {label:<27}{value:>16,}{100*value/nl_ops:>8.2f}%")
    print(f"  {'NONLINEAR TOTAL':<32}{nl_ops:>16,}")
    print()
    print(f"  multiplications per nonlinear invocation : {mult_ops/nl_ops:,.0f}x")
    print(f"  nonlinear share of all invocations       : {100*nl_ops/(mult_ops+nl_ops):.4f}%")
    print()
    exact_left = sum(counts.exact_remaining.values())
    if not exact_left:
        print("  exact float multiplies still present         : none")
        print("  (accumulation stays exact addition; B-PLA replaces multiplies)")
        print()
    if exact_left:
        print("  exact float multiplies still present (not converted):")
        for label, value in counts.exact_remaining.items():
            print(f"    {label:<38}{value:>14,}{100*value/mult_ops:>8.4f}% of multiplies")
        print()
    print(f"  shift-add terms, multiplier T={mult_terms:<12}: {mult_cost:>18,}")
    print(f"  shift-add terms, nonlinear T={nonlinear_terms:<12}: {nl_cost:>18,}")
    print(f"  nonlinear share of shift-add budget      : {100*nl_cost/total:.4f}%")
    print(f"  cost of raising nonlinear T={baseline_terms} -> {nonlinear_terms}      : "
          f"+{100*(nl_cost-nl_cost_baseline)/(mult_cost+nl_cost_baseline):.4f}% of total terms")
    return {
        "model": name,
        "multiplication_invocations": mult_ops,
        "nonlinear_invocations": nl_ops,
        "multiplications_per_nonlinear": mult_ops / nl_ops,
        "multiplier_terms": mult_terms,
        "nonlinear_terms": nonlinear_terms,
        "multiplication_shift_adds": mult_cost,
        "nonlinear_shift_adds": nl_cost,
        "nonlinear_share_of_budget": nl_cost / total,
        "cost_of_raising_nonlinear_terms": (nl_cost - nl_cost_baseline) / (mult_cost + nl_cost_baseline),
        "exact_multiplies_remaining": counts.exact_remaining,
        "breakdown": {"multiplications": counts.multiplications, "nonlinear": counts.nonlinear},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiplication vs nonlinear invocation budget.")
    parser.add_argument("--gpt2-sequence-length", type=int, default=256)
    parser.add_argument("--vit-tokens", type=int, default=197)
    parser.add_argument("--multiplier-terms", type=int, default=2)
    parser.add_argument("--nonlinear-terms", type=int, default=4)
    parser.add_argument(
        "--baseline-nonlinear-terms",
        type=int,
        default=2,
        help="Budget the nonlinear tables would share with the multiplier, for the delta.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "arithmetic_budget.json"
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    records = [
        report(
            f"GPT-2, {args.gpt2_sequence_length} tokens",
            gpt2_counts(12, 768, 12, 3072, args.gpt2_sequence_length, 50257),
            args.multiplier_terms,
            args.nonlinear_terms,
            args.baseline_nonlinear_terms,
        ),
        report(
            f"ViT-Base, {args.vit_tokens} tokens",
            vit_counts(12, 768, 12, 3072, args.vit_tokens, 1000, 16, 3),
            args.multiplier_terms,
            args.nonlinear_terms,
            args.baseline_nonlinear_terms,
        ),
    ]

    args.output.write_text(
        json.dumps(
            {
                "notes": [
                    "Counts follow bpla_softmax_torch and bpla_layer_norm_torch operation by "
                    "operation, so composition-internal multiplies are attributed to the "
                    "multiplier rather than to the nonlinear path.",
                    "Shift-add costs assume the separable multiplier (2*T per product) and the "
                    "one-dimensional tables (T per lookup).",
                    "Arithmetic counts only. Not an energy or area claim.",
                ],
                "configuration": vars(args) | {"output": str(args.output)},
                "models": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
