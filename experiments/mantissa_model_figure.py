"""
What narrowing the mantissa datapath costs, measured inside the models.

The primitive sweep (``mantissa_width_sweep.py``) established that most of
B-PLA's per-product energy goes into additions whose width was inherited from
float32 rather than required by the method, and that narrowing that width is
cheap on sampled operand pairs. This figure asks the question that actually
decides the cost argument: whether the same holds once the error has to survive
48 matmuls, a residual stream, and a vocabulary projection.

It does, and by a wider margin than the primitive sweep suggested. At 12 bits
the per-product NRMSE is 12% worse than at 24, but the model's argmax agreement
is *identical* -- the extra error does not reach the prediction. ViT holds
perfect agreement down to a 10-bit datapath.

Why both metrics are plotted
----------------------------
Agreement alone would overclaim. It is flat where NRMSE is not, and a reader is
entitled to see that the output error does grow even while the decision does
not; a figure showing only the flat curve would be hiding the mechanism. The
top row is therefore the error that degrades and the bottom row is the decision
that survives it, on one shared energy axis.

The dashed line in every panel is W8A8 per-token, measured in the same runs
against the same exact reference. It is the comparison the numbers exist for:
B-PLA at 12 bits costs 1.78x an int8 multiply and stays roughly two orders of
magnitude closer to the exact model.

Caveats
-------
Energy is the 45 nm arithmetic proxy and excludes coefficient-table traffic,
which B-PLA has and an int8 multiplier does not. Narrowing also costs B-PLA its
exactness on powers of two, which no plot here shows -- see the term-budget
discussion.

Usage
-----
    bash run_mantissa.sh                              # produces the JSONs
    python experiments/mantissa_model_figure.py       # produces the figure
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.compute_energy import (
    BPLAComputeConfig,
    ComputeEnergyTablePJ,
    bpla_multiplier_energy_pj,
)

COLOR_BPLA = "#1f77b4"
COLOR_W8A8 = "#2ca02c"


def _load(prefix: str, directory: Path) -> tuple[list[dict], dict | None, str]:
    """Collect one sweep series, ordered widest datapath first."""

    entries = []
    reference = None
    title = prefix
    for path in sorted(directory.glob(f"{prefix}_*.json")):
        match = re.search(r"_(\d+)\.json$", path.name)
        if match is None:
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        rows = {r["backend"]: r for r in record["results"]}
        if "bpla-dyadic" not in rows:
            continue  # A run still in flight writes its exact row first.
        config = record["configuration"]
        entries.append(
            {
                "bits": int(config["mantissa_bits"]),
                "nrmse": rows["bpla-dyadic"]["logit_nrmse"],
                "agreement": rows["bpla-dyadic"]["argmax_agreement"],
            }
        )
        if reference is None and "ptq-w8a8" in rows:
            reference = {
                "nrmse": rows["ptq-w8a8"]["logit_nrmse"],
                "agreement": rows["ptq-w8a8"]["argmax_agreement"],
            }
        exact = record["results"][0]
        if config["models"] == ["gpt2"]:
            title = f"GPT-2, {exact.get('tokens', 0):,} tokens"
        else:
            title = f"ViT-Base, {exact.get('samples', 0)} images"
    entries.sort(key=lambda e: -e["bits"])
    return entries, reference, title


def render(series: list[tuple[list[dict], dict | None, str]], output: Path) -> Path:
    table = ComputeEnergyTablePJ()
    figure, axes = plt.subplots(2, len(series), figsize=(3.8 * len(series), 4.8), sharex=True)
    if len(series) == 1:
        axes = [[axes[0]], [axes[1]]]

    all_energies: list[float] = []
    for column, (entries, reference, title) in enumerate(series):
        energies = [
            bpla_multiplier_energy_pj(
                BPLAComputeConfig(affine_path="dyadic", dyadic_terms=2, mantissa_bits=e["bits"]),
                table,
            )["total_pj"]
            for e in entries
        ]

        top, bottom = axes[0][column], axes[1][column]

        top.plot([*energies], [e["nrmse"] for e in entries], marker="o", markersize=5,
                 color=COLOR_BPLA, linewidth=1.5, label="B-PLA dyadic $T$=2")
        bottom.plot([*energies], [e["agreement"] for e in entries], marker="o", markersize=5,
                    color=COLOR_BPLA, linewidth=1.5)

        for energy, entry in zip(energies, entries):
            top.annotate(str(entry["bits"]), (energy, entry["nrmse"]),
                         textcoords="offset points", xytext=(0, 7), ha="center",
                         fontsize=6.5, color=COLOR_BPLA)

        if reference is not None:
            top.axhline(reference["nrmse"], color=COLOR_W8A8, linestyle="--", linewidth=1.3,
                        label="W8A8 per-token")
            bottom.axhline(reference["agreement"], color=COLOR_W8A8, linestyle="--", linewidth=1.3)
            bottom.annotate(f"W8A8 per-token  {reference['agreement']:.2f}%",
                            xy=(0.97, reference["agreement"]), xycoords=("axes fraction", "data"),
                            ha="right", va="bottom", fontsize=6.8, color=COLOR_W8A8)

        top.set_yscale("log")
        # Headroom above the W8A8 line, which otherwise sits on the panel edge
        # and reads as an axis rather than as the comparison.
        ceiling = max([e["nrmse"] for e in entries]
                      + ([reference["nrmse"]] if reference else []))
        top.set_ylim(min(e["nrmse"] for e in entries) * 0.5, ceiling * 3.0)
        if reference is not None:
            top.annotate(f"W8A8 per-token  {reference['nrmse']:.2e}",
                         xy=(0.97, reference["nrmse"]), xycoords=("axes fraction", "data"),
                         ha="right", va="bottom", fontsize=6.8, color=COLOR_W8A8)
        top.grid(alpha=0.3, which="both")
        top.set_title(title, fontsize=9)
        bottom.grid(alpha=0.3)
        bottom.set_xlabel("arithmetic energy per product (pJ, 45 nm)", fontsize=8.5)
        all_energies.extend(energies)

        # Agreement lives in the top few percent, so a full 0-100 axis would
        # compress every difference that matters into one pixel band. The floor
        # is set below the W8A8 reference so the gap it has to clear is visible.
        floor = min([e["agreement"] for e in entries]
                    + ([reference["agreement"]] if reference else [100.0]))
        bottom.set_ylim(floor - 2.0, 100.6)

        secondary = top.secondary_xaxis(
            "top", functions=(lambda e: e / table.int8_mul, lambda m: m * table.int8_mul)
        )
        secondary.set_xticks([1, 2, 3])
        secondary.set_xticklabels(["1x", "2x", "3x"], fontsize=7)
        secondary.tick_params(length=2)

        if column == 0:
            # Short labels: the long forms collided across the two rows.
            top.set_ylabel("logit NRMSE", fontsize=8.5)
            bottom.set_ylabel("agreement (%)", fontsize=8.5)

    # The axes share x, so a per-column limit would silently clip whichever
    # column reaches further -- as it did when one sweep had a narrower
    # datapath than the other.
    axes[0][0].set_xlim(min(all_energies) - 0.04, max(all_energies) + 0.04)

    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, fontsize=7.5,
                  frameon=False, bbox_to_anchor=(0.5, -0.01))
    figure.suptitle(
        "Narrowing the B-PLA mantissa datapath inside the models\n"
        "numerals: datapath width in bits; top axis: multiples of an int8 multiply",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.93))
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    default_dir = Path(__file__).resolve().parent / "gpu_results"
    parser = argparse.ArgumentParser(description="Render the model-level mantissa sweep.")
    parser.add_argument("--results", type=Path, default=default_dir)
    parser.add_argument("--prefixes", nargs="+", default=["w1", "w2"])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "fig_mantissa_model.png",
    )
    args = parser.parse_args()

    series = [_load(prefix, args.results) for prefix in args.prefixes]
    series = [s for s in series if s[0]]
    if not series:
        raise SystemExit(f"no sweep results found under {args.results}")
    print(f"wrote {render(series, args.output)}")

    for entries, reference, title in series:
        if reference is None:
            continue
        widest = entries[0]
        for entry in entries:
            ratio = reference["nrmse"] / entry["nrmse"]
            same = "same as full width" if entry["agreement"] == widest["agreement"] else ""
            print(
                f"  [{title}] {entry['bits']:2d} bits: agree={entry['agreement']:7.3f}% "
                f"({ratio:5.1f}x closer than W8A8) {same}"
            )


if __name__ == "__main__":
    main()
