"""
Accuracy against arithmetic energy for the multiplier, as the mantissa narrows.

Why this experiment exists
--------------------------
The model-level comparison put B-PLA two orders of magnitude closer to the exact
model than int8 post-training quantization, and the cost model put it 3.3x above
int8 in energy per product. That is the honest headline, and it invites one
obvious question: where does B-PLA's energy actually go, and is any of it
recoverable?

The answer from ``modules/compute_energy.py`` is that at T=2 the multiplier
spends 92% of its energy on 3T+2 fixed-point additions, whose cost is linear in
the mantissa datapath width. That width is 24 bits only because float32's
significand is 24 bits -- nothing about the method requires it. This sweep asks
what accuracy costs at narrower widths, and puts B-PLA, PAM and an int8
multiplier on one accuracy-versus-energy plane so the trade is visible rather
than asserted.

What narrowing gives up
-----------------------
At full width B-PLA approximates only the interaction term ``m1*m2`` and carries
the operand mantissas exactly. Below it, the operands are rounded too. That is a
genuine weakening of the method's claim and is reported as one: the point of the
sweep is to find where the knee is, not to assume there is a free lunch.

Baselines on the same axis
--------------------------
* **PAM** discards ``m1*m2`` entirely; two integer additions, 0.200 pJ.
* **int8** rounds both operands to a symmetric 8-bit grid and multiplies them
  exactly. Its error is entirely operand quantization, where B-PLA's is entirely
  interaction-term residual -- opposite failure modes at comparable cost, which
  is the comparison the paper needs to make.

The int8 operand range is calibrated per distribution with the same min-max rule
the W8A8 backend uses, so it is not handicapped by a range chosen for B-PLA.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.compute_energy import (
    BPLAComputeConfig,
    ComputeEnergyTablePJ,
    bpla_multiplier_energy_pj,
)
from modules.torch_bpla import SharedBPLATables, TorchBPLAConfig, bpla_multiply_torch
from modules.torch_pao import pao_multiply_torch


def sample_operands(
    distribution: str, count: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Operand pairs for the regimes the multiplier has to survive."""

    if distribution == "uniform":
        a = torch.empty(count).uniform_(-6.0, 6.0, generator=generator)
        b = torch.empty(count).uniform_(-6.0, 6.0, generator=generator)
    elif distribution == "normal":
        a = torch.randn(count, generator=generator)
        b = torch.randn(count, generator=generator)
    elif distribution == "lognormal":
        # Weight-times-activation products span decades; a single scale would
        # flatter any method whose error is relative.
        a = torch.randn(count, generator=generator).exp() * torch.randn(
            count, generator=generator
        ).sign()
        b = torch.randn(count, generator=generator).exp() * torch.randn(
            count, generator=generator
        ).sign()
    else:
        raise ValueError(f"Unknown distribution {distribution!r}.")
    return a, b


def relative_rmse(approximate: torch.Tensor, exact: torch.Tensor) -> float:
    return float(approximate.sub(exact).pow(2).mean().sqrt() / exact.pow(2).mean().sqrt())


def gain(approximate: torch.Tensor, exact: torch.Tensor) -> float:
    """Least-squares slope: the part of the error that survives accumulation."""

    return float((approximate * exact).sum() / exact.pow(2).sum())


def int8_multiply(a: torch.Tensor, b: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Quantize both operands to a symmetric grid, then multiply exactly.

    This is the multiplier a W8A8 pipeline uses. Both operands get their own
    min-max range, matching per-tensor activation and per-channel weight
    calibration; the product itself is exact in int32.
    """

    qmax = float(2 ** (bits - 1) - 1)
    scale_a = (a.abs().amax() / qmax).clamp(min=1e-12)
    scale_b = (b.abs().amax() / qmax).clamp(min=1e-12)
    qa = torch.clamp(torch.round(a / scale_a), -qmax, qmax) * scale_a
    qb = torch.clamp(torch.round(b / scale_b), -qmax, qmax) * scale_b
    return qa * qb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=2_000_000)
    parser.add_argument(
        "--mantissa-bits", type=int, nargs="+", default=[24, 20, 16, 14, 12, 10, 8, 6]
    )
    parser.add_argument("--dyadic-terms", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--prefix-bits", type=int, default=4)
    parser.add_argument(
        "--distributions", nargs="+", default=["uniform", "normal", "lognormal"]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "mantissa_width_sweep.json"
    )
    args = parser.parse_args()

    table = ComputeEnergyTablePJ()
    record: dict[str, object] = {
        "configuration": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "energy_table_pj": {
            "fp32_mul": table.fp32_mul,
            "int8_mul": table.int8_mul,
            "int32_add": table.int32_add,
        },
        "notes": [
            "Energy is the 45 nm arithmetic proxy from modules/compute_energy.py; "
            "it excludes table, register and memory traffic, which B-PLA has and "
            "an int8 multiplier does not, so the B-PLA figures are optimistic.",
            "Narrowing the mantissa rounds the operand fractions too, so below "
            "full width B-PLA no longer carries m1 and m2 exactly.",
            "int8 error is operand quantization only; the int32 product is exact.",
        ],
        "results": [],
    }
    results: list[dict[str, object]] = record["results"]  # type: ignore[assignment]

    for distribution in args.distributions:
        generator = torch.Generator().manual_seed(args.seed)
        a, b = sample_operands(distribution, args.num_samples, generator)
        exact = a.double() * b.double()

        pam = pao_multiply_torch(a, b).double()
        results.append(
            {
                "distribution": distribution,
                "method": "pam",
                "mantissa_bits": None,
                "dyadic_terms": None,
                "relative_rmse": relative_rmse(pam, exact),
                "gain": gain(pam, exact),
                "energy_pj": 2 * table.int32_add,
            }
        )
        results.append(
            {
                "distribution": distribution,
                "method": "pam-alpha",
                "mantissa_bits": None,
                "dyadic_terms": None,
                "relative_rmse": relative_rmse(
                    pao_multiply_torch(a, b, _alpha_config()).double(), exact
                ),
                "gain": gain(pao_multiply_torch(a, b, _alpha_config()).double(), exact),
                "energy_pj": 4 * table.int32_add,
            }
        )
        int8 = int8_multiply(a, b).double()
        results.append(
            {
                "distribution": distribution,
                "method": "int8",
                "mantissa_bits": 8,
                "dyadic_terms": None,
                "relative_rmse": relative_rmse(int8, exact),
                "gain": gain(int8, exact),
                "energy_pj": table.int8_mul,
            }
        )

        for terms in args.dyadic_terms:
            for bits in args.mantissa_bits:
                config = TorchBPLAConfig(
                    prefix_bits=args.prefix_bits,
                    affine_path="dyadic",
                    dyadic_terms=terms,
                    mantissa_bits=bits,
                )
                approximate = bpla_multiply_torch(
                    a, b, config, SharedBPLATables(config)
                ).double()
                energy = bpla_multiplier_energy_pj(
                    BPLAComputeConfig(
                        affine_path="dyadic", dyadic_terms=terms, mantissa_bits=bits
                    ),
                    table,
                )
                results.append(
                    {
                        "distribution": distribution,
                        "method": "bpla-dyadic",
                        "mantissa_bits": bits,
                        "dyadic_terms": terms,
                        "relative_rmse": relative_rmse(approximate, exact),
                        "gain": gain(approximate, exact),
                        "energy_pj": energy["total_pj"],
                        "energy_vs_int8": energy["total_pj"] / table.int8_mul,
                    }
                )
                print(
                    f"[{distribution}] T={terms} mantissa={bits:2d}  "
                    f"rel_rmse={results[-1]['relative_rmse']:.4e}  "
                    f"gain={results[-1]['gain']:.6f}  "
                    f"{energy['total_pj']:.3f} pJ "
                    f"({energy['total_pj'] / table.int8_mul:.2f}x int8)",
                    flush=True,
                )
        args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")

    _print_pareto(results, table)
    args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


def _alpha_config():
    from modules.torch_pao import TorchPAOConfig

    return TorchPAOConfig(alpha=1.056)


def _print_pareto(results: list[dict], table: ComputeEnergyTablePJ) -> None:
    """Which configurations are not beaten on both axes at once."""

    for distribution in sorted({str(r["distribution"]) for r in results}):
        rows = [r for r in results if r["distribution"] == distribution]
        frontier = [
            r
            for r in rows
            if not any(
                other["energy_pj"] <= r["energy_pj"]
                and other["relative_rmse"] < r["relative_rmse"]
                for other in rows
            )
        ]
        frontier.sort(key=lambda r: r["energy_pj"])
        print(f"\n=== {distribution}: accuracy-energy frontier ===")
        for r in frontier:
            label = str(r["method"])
            if r["method"] == "bpla-dyadic":
                label += f" T={r['dyadic_terms']} m={r['mantissa_bits']}"
            print(
                f"  {label:26s} {r['energy_pj']:6.3f} pJ "
                f"({r['energy_pj'] / table.int8_mul:5.2f}x int8)  "
                f"rel_rmse={r['relative_rmse']:.4e}"
            )


if __name__ == "__main__":
    main()
