"""
Where B-PLA sits among training-free and integer-only Transformer methods.

The comparison tables in this paper put methods side by side that do not convert
the same operators, and a number read out of such a table invites the wrong
conclusion: a coverage difference looks like a fidelity difference. This figure
exists so the tables can be read correctly. It is a schematic, not a
measurement -- every cell is a property of the cited method, not something we
ran.

The three attributes that matter
--------------------------------
1. **Does it need training?** PAO trains with the approximation in the loop;
   I-BERT and I-ViT both quantize a pretrained model and then fine-tune with
   straight-through estimation. Only W8A8 PTQ, FQ-ViT and B-PLA convert a
   checkpoint with forward calibration alone, which is the setting this paper
   works in.
2. **What does the multiplication become?** This is the axis the paper turns on.
   Every integer-only method above keeps an int8 multiplier array; their dyadic
   arithmetic rescales *between* layers rather than replacing the product. PAM
   turns the product into integer addition. B-PLA turns it into shift-add.
3. **Which nonlinear operators are converted?** W8A8 leaves all three in
   floating point. FQ-ViT reaches Softmax and LayerNorm but never addresses
   GELU -- the paper does not mention it. I-BERT and I-ViT reach all three, at
   the cost of fine-tuning.

The empty cell
--------------
Reading the training-free block: no prior method both removes the multiplier and
converts the nonlinear operators. That intersection is what B-PLA occupies, and
stating it this way is narrower and more defensible than claiming novelty for
any single mechanism -- bit-shift nonlinear approximation, GELU via the sigmoid
form, and power-of-two constants are all prior art, as the Related Work says.

Usage
-----
    python experiments/coverage_matrix_figure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

COLOR_MULTIPLIER = {
    "int8 multiplier": "#2ca02c",
    "integer addition": "#d62728",
    "shift-add": "#1f77b4",
}
COLOR_COVERED = "#333333"
COLOR_ABSENT = "#cccccc"

#: (method, training-free, multiplier, GELU, Softmax, LayerNorm, domain)
#: Coverage states: True covered, False not covered, "ours" covered only by a
#: construction of ours rather than by the cited work.
METHODS = [
    ("W8A8 PTQ", True, "int8 multiplier", False, False, False, "general"),
    ("FQ-ViT", True, "int8 multiplier", False, True, True, "vision"),
    ("B-PLA (this work)", True, "shift-add", True, True, True, "both"),
    ("I-BERT", False, "int8 multiplier", True, True, True, "language"),
    ("I-ViT", False, "int8 multiplier", True, True, True, "vision"),
    ("PAO / PAM", False, "integer addition", "ours", True, True, "both"),
]
OPERATORS = ["GELU", "Softmax", "LayerNorm"]


def render(output: Path) -> Path:
    figure, axis = plt.subplots(figsize=(7.2, 3.4))

    training_free = [m for m in METHODS if m[1]]
    trained = [m for m in METHODS if not m[1]]
    ordered = training_free + trained
    rows = len(ordered)

    # Columns: multiplier arithmetic, then one per nonlinear operator.
    x_multiplier = 0.0
    x_operators = [1.5, 2.3, 3.1]
    x_domain = 4.0

    # Band behind the training-free block: the figure's whole argument is about
    # what is and is not present inside it.
    band_left, band_right = -3.15, x_domain + 0.5
    axis.add_patch(
        Rectangle(
            (band_left, rows - len(training_free) - 0.5),
            band_right - band_left,
            len(training_free),
            facecolor="#eef3fa",
            edgecolor="none",
            zorder=0,
        )
    )
    # Group labels sit to the left of the method names: on the right they ran
    # into the model-domain column.
    axis.annotate(
        "training-free\nforward calibration only",
        xy=(band_left + 0.12, rows - len(training_free) / 2 - 0.5),
        ha="center",
        va="center",
        rotation=90,
        fontsize=7,
        color="#31577f",
        fontweight="bold",
    )
    axis.annotate(
        "requires training",
        xy=(band_left + 0.12, (len(trained) - 1) / 2),
        ha="center",
        va="center",
        rotation=90,
        fontsize=7,
        color="0.45",
    )

    for index, (name, _free, multiplier, gelu, softmax, layernorm, domain) in enumerate(ordered):
        y = rows - 1 - index
        weight = "bold" if name.startswith("B-PLA") else "normal"
        axis.annotate(
            name,
            xy=(-0.82, y),
            ha="right",
            va="center",
            fontsize=8,
            fontweight=weight,
        )

        colour = COLOR_MULTIPLIER[multiplier]
        axis.annotate(
            multiplier,
            xy=(x_multiplier, y),
            ha="center",
            va="center",
            fontsize=7.5,
            color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=colour, edgecolor="none"),
        )

        for position, covered in zip(x_operators, (gelu, softmax, layernorm)):
            if covered == "ours":
                # PAO's models use ReLU and the paper never defines a piecewise
                # affine GELU; ours is a construction, and marking it as theirs
                # would credit them with something they did not claim.
                axis.scatter(
                    [position], [y], s=95, facecolors="none",
                    edgecolors=COLOR_COVERED, linewidths=1.3, zorder=3,
                )
                axis.annotate(
                    "*", xy=(position, y), ha="center", va="center",
                    fontsize=9, color=COLOR_COVERED, zorder=4,
                )
            elif covered:
                axis.scatter([position], [y], s=95, color=COLOR_COVERED, zorder=3)
            else:
                axis.scatter(
                    [position], [y], s=95, facecolors="none",
                    edgecolors=COLOR_ABSENT, linewidths=1.3, zorder=3,
                )

        axis.annotate(domain, xy=(x_domain, y), ha="center", va="center", fontsize=7.5, color="0.35")

    axis.annotate(
        "multiplication becomes",
        xy=(x_multiplier, rows - 0.35),
        ha="center", va="bottom", fontsize=8, fontweight="bold",
    )
    axis.annotate(
        "nonlinear operators converted",
        xy=(sum(x_operators) / len(x_operators), rows - 0.35),
        ha="center", va="bottom", fontsize=8, fontweight="bold",
    )
    for position, label in zip(x_operators, OPERATORS):
        axis.annotate(
            label, xy=(position, rows - 0.62), ha="center", va="bottom", fontsize=7.5
        )
    axis.annotate(
        "models", xy=(x_domain, rows - 0.62), ha="center", va="bottom", fontsize=7.5
    )

    axis.annotate(
        "$*$ our construction; the cited work uses ReLU and defines no piecewise affine GELU",
        xy=(band_left, -0.95), ha="left", va="center", fontsize=6.5, color="0.4",
    )

    axis.set_xlim(band_left - 0.15, band_right)
    axis.set_ylim(-1.2, rows + 0.3)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the coverage schematic.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "fig_coverage_matrix.png",
    )
    args = parser.parse_args()
    print(f"wrote {render(args.output)}")


if __name__ == "__main__":
    main()
