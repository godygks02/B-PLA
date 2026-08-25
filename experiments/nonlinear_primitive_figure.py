"""
Nonlinear-primitive approximation figure for the submission.

The submission plan rules out the usual "exact and approximate curves drawn on
top of each other" plot: at these error levels the two curves are visually
identical and the figure says nothing. This script instead draws, on one shared
x-axis:

  1. exact vs. B-PLA output with the FP-field routing boundaries marked;
  2. signed error, which is what the overlaid curves hide;
  3. the activation density actually observed in the pretrained model, so a
     reader can tell whether the error lives where the inputs live.

The activation histogram is measured from the real checkpoint by default
(``--model vit``); pass ``--model none`` to fall back to a standard normal and
have the figure say so.

Also emits a per-primitive fidelity table (exp / reciprocal / rsqrt / GELU)
weighted by that empirical distribution, which is Table 2 of the plan.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.torch_bpla import (
    TARGETS,
    SharedBPLATables,
    TorchBPLAConfig,
    activation_prefix_index_torch,
    bpla_activation_torch,
    build_activation_table_torch,
    _functional_bpla,
)
from modules.torch_pao import (
    TorchPAOConfig,
    paexp_torch,
    pao_divide_torch,
    pao_gelu_torch,
    pasqrt_torch,
)


def collect_gelu_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, str]:
    """Record the real GELU inputs of the pretrained model, if available."""

    if args.model == "none":
        return torch.randn(200000) * 2.0, "standard normal (no model)"

    from datasets import load_dataset
    from transformers import ViTForImageClassification, ViTImageProcessor

    model = ViTForImageClassification.from_pretrained(args.vit_model_id).eval()
    processor = ViTImageProcessor.from_pretrained(args.vit_model_id)
    dataset = load_dataset(args.vit_dataset_id)
    split = dataset.get("validation") or dataset.get("test") or dataset["train"]

    observed: list[torch.Tensor] = []

    def hook(_module, inputs):
        if inputs and isinstance(inputs[0], torch.Tensor):
            flat = inputs[0].detach().flatten()
            # Subsample: the full tensor is millions of values per batch.
            step = max(1, flat.numel() // 40000)
            observed.append(flat[::step].clone())

    handles = []
    for child in model.modules():
        name = child.__class__.__name__.lower()
        if isinstance(child, torch.nn.GELU) or "geluactivation" in name:
            handles.append(child.register_forward_pre_hook(hook))
        activation = getattr(child, "intermediate_act_fn", None)
        if activation is not None and not isinstance(activation, torch.nn.Module):
            original = activation

            def wrapper(x, _original=original):
                hook(None, (x,))
                return _original(x)

            child.intermediate_act_fn = wrapper

    images = [split[i]["image"].convert("RGB") for i in range(args.num_images)]
    with torch.no_grad():
        model(**processor(images, return_tensors="pt"))
    for handle in handles:
        handle.remove()

    return torch.cat(observed), f"{args.vit_model_id} on {args.num_images} images"


def segment_boundaries(config: TorchBPLAConfig, table: dict) -> torch.Tensor:
    """x-positions where the FP-field router switches segment."""

    grid = torch.linspace(float(table["x_min"]), float(table["x_max"]), 200000)
    index = activation_prefix_index_torch(
        grid, config, int(table["min_e_routing"]), int(table["max_e_routing"])
    )
    changes = (index[1:] != index[:-1]).nonzero().flatten()
    return grid[changes]


def primitive_table(args: argparse.Namespace, samples: torch.Tensor) -> list[dict[str, object]]:
    """Fidelity of every supported primitive, on its own domain.

    Pointwise relative error is not usable here: GELU crosses zero, so a
    relative error is unbounded at points where nothing is actually wrong. We
    report absolute error together with NRMSE, the RMS error divided by the RMS
    of the exact output, which is scale-free and finite through a zero crossing.
    """

    pao_config = TorchPAOConfig()
    rows: list[dict[str, object]] = []

    def record(
        name: str,
        method: str,
        terms: int | None,
        approx: torch.Tensor,
        exact: torch.Tensor,
        domain: str,
    ) -> None:
        error = approx - exact
        rows.append(
            {
                "primitive": name,
                "method": method,
                "dyadic_terms": terms,
                "domain": domain,
                "mae": float(error.abs().mean()),
                "rmse": float(error.pow(2).mean().sqrt()),
                "nrmse": float(error.pow(2).mean().sqrt() / exact.pow(2).mean().sqrt()),
                "p99_abs_error": float(torch.quantile(error.abs(), 0.99)),
                "max_abs_error": float(error.abs().max()),
            }
        )

    # GELU on the model's own activation distribution, which is the only
    # domain where its error actually matters.
    gelu_exact = TARGETS["gelu"](samples)
    record("gelu", "pao", None, pao_gelu_torch(samples, pao_config), gelu_exact, "empirical activations")
    for terms in [None] + list(args.dyadic_term_sweep):
        config = TorchBPLAConfig(
            prefix_bits=args.prefix_bits,
            affine_path="float" if terms is None else "dyadic",
            dyadic_terms=terms or 1,
            max_shift=args.max_shift,
            activation_range=args.activation_range,
        )
        table = build_activation_table_torch("gelu", config, samples.device, samples.dtype)
        record(
            "gelu",
            "bpla-float" if terms is None else "bpla-dyadic",
            terms,
            bpla_activation_torch(samples, table, config),
            gelu_exact,
            "empirical activations",
        )

    # The composite Softmax/LayerNorm primitives, each on the domain the
    # composition actually presents to them.
    functional_domains = {
        "exp2_fraction": (torch.rand(200000), torch.exp2, "mantissa fraction [0,1)"),
        "reciprocal_unit_mantissa": (
            1.0 + torch.rand(200000),
            torch.reciprocal,
            "unit mantissa [1,2)",
        ),
        "rsqrt_mantissa": (0.5 + 1.5 * torch.rand(200000), torch.rsqrt, "mantissa [0.5,2)"),
    }
    pao_equivalent = {
        "exp2_fraction": lambda x: paexp_torch(x * float(torch.log(torch.tensor(2.0))), pao_config),
        "reciprocal_unit_mantissa": lambda x: pao_divide_torch(torch.ones_like(x), x, pao_config),
        "rsqrt_mantissa": lambda x: pao_divide_torch(
            torch.ones_like(x), pasqrt_torch(x, pao_config), pao_config
        ),
    }
    for name, (inputs, exact_fn, domain) in functional_domains.items():
        exact = exact_fn(inputs)
        record(name, "pao", None, pao_equivalent[name](inputs), exact, domain)
        for terms in [None] + list(args.dyadic_term_sweep):
            config = TorchBPLAConfig(
                prefix_bits=args.prefix_bits,
                affine_path="float" if terms is None else "dyadic",
                dyadic_terms=terms or 1,
                max_shift=args.max_shift,
            )
            tables = SharedBPLATables(config)
            record(
                name,
                "bpla-float" if terms is None else "bpla-dyadic",
                terms,
                _functional_bpla(inputs, name, config, tables),
                exact,
                domain,
            )
    return rows


def calibration_policy_table(
    args: argparse.Namespace, samples: torch.Tensor
) -> list[dict[str, object]]:
    """Compare range policies for the shared GELU table.

    The probes calibrate on the maximum observed magnitude. For a distribution
    with a long thin tail that spends most of the table on inputs that almost
    never occur, so we also measure percentile rules, which clamp the tail
    instead. Clamping is not free -- it introduces a hard error on the values it
    truncates -- so the question is empirical.
    """

    exact = TARGETS["gelu"](samples)
    rows: list[dict[str, object]] = []
    policies: list[tuple[str, float]] = [("max", float(samples.abs().max()))]
    for quantile in args.calibration_quantiles:
        policies.append((f"p{100 * quantile:g}", float(torch.quantile(samples.abs(), quantile))))

    for name, limit in policies:
        for terms in (2, 4):
            config = TorchBPLAConfig(
                prefix_bits=args.prefix_bits,
                affine_path="dyadic",
                dyadic_terms=terms,
                max_shift=args.max_shift,
                activation_range=limit,
            )
            table = build_activation_table_torch("gelu", config, samples.device, samples.dtype)
            error = bpla_activation_torch(samples, table, config) - exact
            rows.append(
                {
                    "policy": name,
                    "range": limit,
                    "dyadic_terms": terms,
                    "clamped_fraction": float((samples.abs() > limit).float().mean()),
                    "nrmse": float(error.pow(2).mean().sqrt() / exact.pow(2).mean().sqrt()),
                    "max_abs_error": float(error.abs().max()),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Nonlinear primitive figure and table.")
    parser.add_argument("--model", choices=["vit", "none"], default="vit")
    parser.add_argument("--vit-model-id", default="google/vit-base-patch16-224")
    parser.add_argument("--vit-dataset-id", default="johnowhitaker/imagenette2-320")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--prefix-bits", type=int, default=4)
    parser.add_argument("--dyadic-terms", type=int, default=2)
    parser.add_argument("--dyadic-term-sweep", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    parser.add_argument("--plot-quantile", type=float, default=0.99)
    parser.add_argument("--calibration-quantiles", type=float, nargs="+", default=[0.9999, 0.999, 0.99])
    parser.add_argument("--max-shift", type=int, default=16)
    parser.add_argument("--activation-range", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    samples, provenance = collect_gelu_inputs(args)
    samples = samples.float()
    if args.activation_range is None:
        # Same calibration rule the probes use: one symmetric range from the
        # observed maximum magnitude.
        args.activation_range = float(samples.abs().max())
    print(f"activation source: {provenance}")
    print(f"observed |x| max : {samples.abs().max():.4f}  (calibrated range)")
    print(f"samples          : {samples.numel():,}")

    rows = primitive_table(args, samples)
    calibration = calibration_policy_table(args, samples)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "activation_source": provenance,
        "calibrated_range": args.activation_range,
        "prefix_bits": args.prefix_bits,
        "dyadic_terms": args.dyadic_terms,
        "max_shift": args.max_shift,
        "sample_count": int(samples.numel()),
        "notes": [
            "GELU is evaluated on the pretrained model's own activation distribution, "
            "not on a uniform grid, because that is where the error matters.",
            "The PAO GELU is our composition from PA primitives; the original paper "
            "uses ReLU and defines no piecewise-affine GELU.",
        ],
        "primitives": rows,
        "calibration_policies": calibration,
    }
    (output_dir / "nonlinear_primitive_results.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )

    header = f"{'primitive':<26}{'method':<13}{'T':>3}{'MAE':>12}{'NRMSE':>12}{'max |err|':>12}   vs PAO"
    print()
    print(header)
    print("-" * len(header))
    baseline = {r["primitive"]: r["nrmse"] for r in rows if r["method"] == "pao"}
    for row in rows:
        ratio = baseline[row["primitive"]] / row["nrmse"] if row["nrmse"] else float("inf")
        verdict = "-" if row["method"] == "pao" else (f"{ratio:.1f}x better" if ratio >= 1 else f"{1/ratio:.1f}x worse")
        print(
            f"{row['primitive']:<26}{row['method']:<13}"
            f"{row['dyadic_terms'] if row['dyadic_terms'] else '-':>3}"
            f"{row['mae']:>12.4e}{row['nrmse']:>12.4e}{row['max_abs_error']:>12.4e}   {verdict}"
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib unavailable; skipping figure.")
        return

    limit = args.activation_range
    # The calibrated range is set by the largest observed activation, but almost
    # none of the distribution lives out there and GELU is affine long before
    # the edge. Plot the region that carries the mass, with the full range as an
    # inset so the tail behaviour is still visible.
    view = float(torch.quantile(samples.abs(), args.plot_quantile))
    grid = torch.linspace(-limit, limit, 20000)
    exact = TARGETS["gelu"](grid)
    curves = {}
    for path in ("float", "dyadic"):
        config = TorchBPLAConfig(
            prefix_bits=args.prefix_bits,
            affine_path=path,
            dyadic_terms=args.dyadic_terms,
            max_shift=args.max_shift,
            activation_range=limit,
        )
        table = build_activation_table_torch("gelu", config, grid.device, grid.dtype)
        curves[f"B-PLA {path}"] = (bpla_activation_torch(grid, table, config), config, table)
    pao_curve = pao_gelu_torch(grid, TorchPAOConfig())

    boundaries = segment_boundaries(curves["B-PLA float"][1], curves["B-PLA float"][2])

    figure, axes = plt.subplots(
        3, 1, figsize=(6.4, 7.2), sharex=True, height_ratios=[2.0, 1.5, 1.1], constrained_layout=True
    )
    top, middle, bottom = axes

    # Drawing all ~350 segment boundaries fills the panel with grey. Shade
    # alternate exponent octaves instead: the octave is the level at which the
    # routing structure actually changes, and each one holds 2^k segments.
    octaves = [float(o) for o in boundaries if abs(math.log2(abs(float(o)) + 1e-12) % 1.0) < 1e-3]
    edges = sorted({0.0} | {abs(o) for o in octaves if abs(o) <= limit} | {limit})
    for index in range(len(edges) - 1):
        if index % 2:
            continue
        for axis in axes:
            for sign in (1, -1):
                axis.axvspan(sign * edges[index], sign * edges[index + 1], color="0.93", zorder=0)

    top.plot(grid, exact, color="black", linewidth=1.6, label="exact GELU", zorder=4)
    top.plot(grid, curves["B-PLA float"][0], color="#1f77b4", linewidth=1.1, label="B-PLA float", zorder=3)
    top.plot(
        grid,
        curves["B-PLA dyadic"][0],
        color="#2ca02c",
        linewidth=1.1,
        linestyle="--",
        label=f"B-PLA dyadic $T$={args.dyadic_terms}",
        zorder=3,
    )
    top.plot(grid, pao_curve, color="#d62728", linewidth=1.0, linestyle=":", label="PAO composition", zorder=2)
    top.set_ylabel("GELU$(x)$")
    top.legend(fontsize=8, loc="upper left")
    top.set_title(
        f"GELU replacement, $k$={args.prefix_bits}, calibrated range $\\pm${limit:.2f}\n"
        f"shading: alternate exponent octaves ($2^k$ routed segments each)",
        fontsize=10,
    )
    top.grid(alpha=0.25)

    # Full calibrated range, where GELU is affine and every method is exact.
    inset = top.inset_axes([0.62, 0.08, 0.36, 0.45])
    inset.plot(grid, exact, color="black", linewidth=1.0)
    inset.plot(grid, curves["B-PLA float"][0], color="#1f77b4", linewidth=0.8)
    inset.plot(grid, pao_curve, color="#d62728", linewidth=0.8, linestyle=":")
    inset.axvspan(-view, view, color="0.85")
    inset.set_xlim(-limit, limit)
    inset.tick_params(labelsize=6)
    inset.set_title("full calibrated range", fontsize=6)

    middle.plot(grid, curves["B-PLA float"][0] - exact, color="#1f77b4", linewidth=0.9)
    middle.plot(grid, curves["B-PLA dyadic"][0] - exact, color="#2ca02c", linewidth=0.9, linestyle="--")
    middle.plot(grid, pao_curve - exact, color="#d62728", linewidth=0.9, linestyle=":")
    middle.axhline(0.0, color="black", linewidth=0.6)
    middle.set_ylabel("signed error")
    middle.set_yscale("symlog", linthresh=1e-5)
    middle.grid(alpha=0.25)

    bottom.hist(
        samples.numpy(),
        bins=600,
        range=(-view, view),
        color="0.4",
        density=True,
    )
    bottom.set_ylabel("density")
    bottom.set_xlabel("$x$")
    bottom.set_yscale("log")
    bottom.set_title(
        f"observed GELU inputs: {provenance}"
        f"  ({100 * args.plot_quantile:.4g}% of |x| below {view:.2f})",
        fontsize=8,
    )
    bottom.grid(alpha=0.25)
    top.set_xlim(-view, view)

    path = output_dir / "fig_gelu_three_panel.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    print(f"\nwrote {path}")
    print(f"wrote {output_dir / 'nonlinear_primitive_results.json'}")


if __name__ == "__main__":
    main()
