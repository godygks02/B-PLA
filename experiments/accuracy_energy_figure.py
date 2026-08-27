"""
The accuracy-versus-arithmetic-energy plane for one scalar product.

This is the figure the paper's central claim rests on. Reporting fidelity alone
would flatter B-PLA, which preserves sign and exponent exactly and approximates
only the mantissa interaction term -- a far less aggressive approximation than
rounding both operands to eight bits. Reporting energy alone would flatter int8.
Putting both on one plane is the only honest way to state the trade, and it also
answers the question a reviewer asks first: if post-training quantization is
cheaper, why would anyone want this?

What the plane shows
--------------------
* **PAM and PAM+alpha are dominated.** PAM's two integer additions cost exactly
  what an int8 multiply costs and buy less accuracy; the alpha correction is a
  second piecewise affine multiply, so it doubles the cost to draw level. A
  multiplication-free method is not automatically a cheap one.
* **B-PLA and int8 both sit on the frontier**, at different points. Narrowing
  the mantissa datapath walks B-PLA down the frontier toward int8's cost, and
  the walk is cheap until the datapath gets close to the operand precision
  itself.
* **The two have opposite failure modes.** int8's error is entirely operand
  quantization; B-PLA's is entirely interaction-term residual. That is why the
  curves cross rather than run parallel.

Reading the axes
----------------
Down and to the left is better. The energy axis is linear because the whole
span is under 5x and the paper quotes ratios; the top axis reads those ratios
off directly in multiples of an int8 multiply. The fp32 multiplier is exact, so
it has no position on a logarithmic error axis, and at 3.70 pJ it is far enough
right that plotting it would compress everything that matters into a corner --
it is named in the title instead.

Caveats that belong in the caption
----------------------------------
The log-normal panel overstates int8's weakness. Both operands there are
quantized with one min-max scale each, while a real W8A8 pipeline uses
per-output-channel weight scales and per-token activation scales, which absorb
most of that spread. It is a stress case for a single shared scale, not a
prediction of deployed W8A8; the model-level agreement figures are the ground
truth and they put W8A8 far closer to B-PLA than this panel does.

The energy model is the 45 nm arithmetic proxy from ``modules/compute_energy.py``
and excludes table, register and memory traffic. B-PLA reads a coefficient table
and int8 and PAM do not, so every B-PLA point here is optimistic by an amount we
have not yet quantified. Shifts are modelled as free, which also favours B-PLA.
These are arithmetic estimates, not synthesis results.

Usage
-----
    python experiments/mantissa_width_sweep.py          # produces the JSON
    python experiments/accuracy_energy_figure.py        # produces the figure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: Kept consistent with fig_multiplier_pareto so the two read as one family.
COLOR_PAM = "#d62728"
COLOR_PAM_ALPHA = "#ff7f0e"
COLOR_INT8 = "#2ca02c"
COLOR_TERMS = {2: "#1f77b4", 3: "#9467bd", 4: "#8c564b", 1: "#17becf"}

DISTRIBUTION_LABEL = {
    "uniform": "uniform operands",
    "normal": "normal operands",
    "lognormal": "log-normal operands",
}


def _frontier(rows: list[dict]) -> list[dict]:
    """Points nothing beats on both axes at once, ordered by energy."""

    keep = [
        row
        for row in rows
        if not any(
            other["energy_pj"] <= row["energy_pj"]
            and other["relative_rmse"] < row["relative_rmse"]
            for other in rows
        )
    ]
    return sorted(keep, key=lambda r: r["energy_pj"])


def _draw(axis, rows: list[dict], int8_pj: float, annotate_widths: set[int]) -> None:
    frontier = _frontier(rows)
    if frontier:
        axis.step(
            [r["energy_pj"] for r in frontier],
            [r["relative_rmse"] for r in frontier],
            where="post",
            color="0.82",
            linewidth=7.0,
            solid_capstyle="butt",
            zorder=1,
            label="Pareto frontier",
        )

    # B-PLA: one curve per term budget, walked by mantissa datapath width.
    for terms in sorted({r["dyadic_terms"] for r in rows if r["method"] == "bpla-dyadic"}):
        points = sorted(
            (r for r in rows if r["method"] == "bpla-dyadic" and r["dyadic_terms"] == terms),
            key=lambda r: r["energy_pj"],
        )
        colour = COLOR_TERMS.get(int(terms), "#1f77b4")
        axis.plot(
            [r["energy_pj"] for r in points],
            [r["relative_rmse"] for r in points],
            marker="o",
            markersize=4.5,
            linewidth=1.4,
            color=colour,
            zorder=3,
            label=f"B-PLA $T$={int(terms)}",
        )
        if int(terms) != 2:
            # Both curves are the same shape in the flat tail and their labels
            # collide there whichever side they take. Naming the widths once, on
            # the recommended term budget, is enough to read either curve.
            continue
        for row in points:
            width = int(row["mantissa_bits"])
            if width not in annotate_widths:
                continue
            axis.annotate(
                str(width),
                (row["energy_pj"], row["relative_rmse"]),
                textcoords="offset points",
                xytext=(2, 7),
                ha="left",
                fontsize=6.5,
                color=colour,
            )

    for method, colour, label, marker in (
        ("int8", COLOR_INT8, "int8 multiplier (W8A8)", "*"),
        ("pam", COLOR_PAM, "PAM", "*"),
        ("pam-alpha", COLOR_PAM_ALPHA, r"PAM $+\alpha$", "*"),
    ):
        point = next((r for r in rows if r["method"] == method), None)
        if point is None:
            continue
        axis.scatter(
            [point["energy_pj"]],
            [point["relative_rmse"]],
            marker=marker,
            s=210,
            color=colour,
            edgecolors="white",
            linewidths=0.6,
            zorder=5,
            label=label,
        )

    axis.set_yscale("log")
    # Linear in energy: the whole span is under 5x, and the paper quotes ratios
    # like "1.78x int8", which a linear axis reads off directly. fp32 sits at
    # 3.70 pJ, far off to the right, and is noted rather than plotted -- giving
    # it an axis position would squeeze every point that matters into a corner.
    axis.set_xlim(0.13, 0.95)
    errors = [r["relative_rmse"] for r in rows]
    axis.set_ylim(min(errors) * 0.55, max(errors) * 2.2)
    axis.grid(alpha=0.3, which="both")

    top = axis.secondary_xaxis("top", functions=(lambda e: e / int8_pj, lambda m: m * int8_pj))
    top.set_xticks([1, 2, 3, 4])
    top.set_xticklabels([f"{m}x" for m in (1, 2, 3, 4)], fontsize=7)
    top.tick_params(length=2)


def render(record: dict, output: Path, distributions: list[str]) -> Path:
    fp32_pj = float(record["energy_table_pj"]["fp32_mul"])
    int8_pj = float(record["energy_table_pj"]["int8_mul"])
    # Widths worth naming: the full-precision end, the knee, and the point where
    # narrowing starts to cost real accuracy. Labelling all eight collides.
    annotate_widths = {24, 16, 12, 10, 8, 6}
    results = record["results"]
    available = [d for d in distributions if any(r["distribution"] == d for r in results)]
    if not available:
        raise SystemExit(f"No requested distribution present in the record: {distributions}")

    figure, axes = plt.subplots(
        1, len(available), figsize=(3.6 * len(available), 3.3), sharey=True
    )
    if len(available) == 1:
        axes = [axes]

    for index, (axis, distribution) in enumerate(zip(axes, available)):
        rows = [r for r in results if r["distribution"] == distribution]
        _draw(axis, rows, int8_pj, annotate_widths)
        axis.set_xlabel("arithmetic energy per product (pJ, 45 nm)")
        axis.set_xticks([0.2, 0.4, 0.6, 0.8])
        if index == 0:
            axis.set_ylabel("relative RMSE of the product")
        axis.set_title(DISTRIBUTION_LABEL.get(distribution, distribution), fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(6, len(labels)),
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    figure.suptitle(
        "Accuracy against arithmetic energy for one scalar product\n"
        f"top axis: multiples of an int8 multiply (fp32 is exact at {fp32_pj:.2f} pJ, off scale); "
        "numerals: B-PLA mantissa datapath width in bits",
        fontsize=9,
        y=1.02,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.97))
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the accuracy-energy plane.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "mantissa_width_sweep.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "fig_accuracy_energy.png",
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=["normal", "lognormal"],
        help="One panel each, in this order. The sweep also records 'uniform', "
             "which tracks 'normal' closely and is omitted to keep the figure "
             "to a two-column width.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"{args.input} not found. Run experiments/mantissa_width_sweep.py first."
        )
    record = json.loads(args.input.read_text(encoding="utf-8"))
    path = render(record, args.output, args.distributions)
    print(f"wrote {path}")

    # The numbers the caption has to quote, so they are never retyped by hand.
    for distribution in args.distributions:
        rows = [r for r in record["results"] if r["distribution"] == distribution]
        if not rows:
            continue
        int8 = next((r for r in rows if r["method"] == "int8"), None)
        if int8 is None:
            continue
        better = [
            r
            for r in rows
            if r["method"] == "bpla-dyadic" and r["relative_rmse"] < int8["relative_rmse"]
        ]
        if not better:
            continue
        cheapest = min(better, key=lambda r: r["energy_pj"])
        print(
            f"  [{distribution}] cheapest B-PLA beating int8: "
            f"T={int(cheapest['dyadic_terms'])}, mantissa={int(cheapest['mantissa_bits'])} bits, "
            f"{cheapest['energy_pj']:.3f} pJ ({cheapest['energy_pj'] / int8['energy_pj']:.2f}x int8), "
            f"{int8['relative_rmse'] / cheapest['relative_rmse']:.1f}x more accurate"
        )


if __name__ == "__main__":
    main()
