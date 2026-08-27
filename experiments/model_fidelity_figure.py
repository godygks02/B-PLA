"""
How much of the exact model each training-free backend preserves.

The model tables carry these numbers, but a table hides the thing that matters
most about them: the separation is two orders of magnitude, and a linear reading
of "0.06 versus 0.0006" does not land the way a log axis does.

Why logit NRMSE and not the task metric
---------------------------------------
Both are reported, but the bars are NRMSE and the annotation is paired argmax
agreement, because the task metric is the one number that can mislead here. On
ViT, PAM scores *above* the exact model (87.89% against 86.72% top-1) while
disagreeing with it on 2.7% of images, and W8A8 reproduces the exact top-1
digit for digit while disagreeing on 2.0%. Averages over a mostly-easy
distribution can improve while the model's actual behaviour drifts. Paired
agreement against the same checkpoint cannot.

The three conditions
--------------------
* **GPT-2, blocks** and **ViT, blocks** are the headline matched runs: every
  backend inserted into one checkpoint, over one sample list, against one exact
  reference.
* **GPT-2, full weighted coverage** additionally converts the vocabulary
  projection, which is 31% of GPT-2's weighted multiplies and is left exact in
  the other two. It separates the methods most sharply: W8A8 falls to 38.6%
  agreement while B-PLA holds 99.9%.

Reading the third panel fairly
------------------------------
That panel is a coverage comparison, not evidence that deployed W8A8 fails.
Quantizing the vocabulary projection is outside the usual W8A8 recipe, which
commonly leaves the output head in higher precision -- and the reason is visible
in our own ablation: with the head converted, quantizing its *weights* alone
costs 84.8% agreement, while quantizing its *input activations* alone costs
39.6%. The projection amplifies an 8-bit relative error on the final hidden
state into the argmax, because the gaps between competing logits are small
against the logits' own scale. It is the same fact that makes row-centering
necessary in the metric.

So the honest claim is about coverage, not about a fair fight the baseline
lost: B-PLA converts every weighted multiply in the model and keeps the
model's behaviour; W8A8 converts the blocks and has to stop there.

Usage
-----
    python experiments/model_fidelity_figure.py
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

#: One colour per method family, consistent with the other figures.
FAMILY_COLOUR = {
    "ptq-w8a8": "#2ca02c",
    "ptq-w8a8-static": "#98df8a",
    "pao": "#d62728",
    "pao-alpha": "#ff7f0e",
    "bpla-dyadic": "#1f77b4",
    "bpla-float": "#aec7e8",
}
LABEL = {
    "ptq-w8a8": "W8A8 per-token",
    "ptq-w8a8-static": "W8A8 per-tensor",
    "pao": "PAM",
    "pao-alpha": r"PAM $+\alpha$",
    "bpla-dyadic": "B-PLA dyadic",
    "bpla-float": "B-PLA float",
}
#: Bottom to top, so the strongest method ends up at the top of each panel.
ORDER = [
    "ptq-w8a8-static",
    "pao",
    "ptq-w8a8",
    "pao-alpha",
    "bpla-float",
    "bpla-dyadic",
]


def _panel_title(record: dict) -> str:
    config = record["configuration"]
    exact = record["results"][0]
    if config["models"] == ["gpt2"]:
        model = config.get("gpt2_model_id", "gpt2")
        scale = f"{exact.get('tokens', 0):,} tokens"
        reference = f"exact PPL {exact['perplexity']:.2f}"
    else:
        model = config.get("vit_model_id", "vit").split("/")[-1]
        scale = f"{exact.get('samples', 0)} images"
        reference = f"exact top-1 {exact['top1']:.2f}%"
    coverage = (
        "blocks + vocabulary projection" if config.get("replace_lm_head") else "blocks only"
    )
    return f"{model}, {coverage}\n{scale}; {reference}"


def _draw(axis, record: dict) -> None:
    rows = {r["backend"]: r for r in record["results"] if r["backend"] != "exact"}
    present = [b for b in ORDER if b in rows]
    positions = list(range(len(present)))

    axis.barh(
        positions,
        [rows[b]["logit_nrmse"] for b in present],
        color=[FAMILY_COLOUR[b] for b in present],
        height=0.68,
    )
    for position, backend in zip(positions, present):
        # Agreement is the figure the text argues from, so it travels with the
        # bar rather than living in a separate panel the reader has to align.
        axis.annotate(
            f"  {rows[backend]['argmax_agreement']:.1f}%",
            (rows[backend]["logit_nrmse"], position),
            va="center",
            fontsize=6.8,
            color="0.2",
        )

    axis.set_yticks(positions)
    axis.set_yticklabels([LABEL[b] for b in present], fontsize=7.5)
    axis.set_xscale("log")
    axis.grid(alpha=0.3, axis="x", which="both")
    axis.set_axisbelow(True)
    axis.set_title(_panel_title(record), fontsize=8.5)


def render(records: list[dict], output: Path) -> Path:
    figure, axes = plt.subplots(
        1, len(records), figsize=(3.6 * len(records), 3.0), sharex=True
    )
    if len(records) == 1:
        axes = [axes]

    for axis, record in zip(axes, records):
        _draw(axis, record)

    # Headroom for the agreement labels, which sit to the right of each bar.
    widest = max(
        r["logit_nrmse"] for record in records for r in record["results"] if "logit_nrmse" in r
    )
    narrowest = min(
        r["logit_nrmse"] for record in records for r in record["results"] if "logit_nrmse" in r
    )
    axes[0].set_xlim(narrowest * 0.45, widest * 9.0)

    figure.supxlabel("row-centered logit NRMSE  (lower is better)", fontsize=9, y=0.02)
    figure.suptitle(
        "Training-free drop-in fidelity; percentages are paired argmax agreement with the exact model",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.93))
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    default_dir = Path(__file__).resolve().parent / "gpu_results"
    parser = argparse.ArgumentParser(description="Render the model-level fidelity figure.")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[
            default_dir / "p1_gpt2_ptq.json",
            default_dir / "p2_vit_ptq.json",
            default_dir / "p5_gpt2_full_coverage.json",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "fig_model_fidelity.png",
    )
    args = parser.parse_args()

    missing = [p for p in args.inputs if not p.exists()]
    if missing:
        raise SystemExit(f"missing result files: {missing}")
    records = [json.loads(p.read_text(encoding="utf-8")) for p in args.inputs]
    print(f"wrote {render(records, args.output)}")

    # The ratios the text quotes, taken from the artifacts rather than retyped.
    for path, record in zip(args.inputs, records):
        rows = {r["backend"]: r for r in record["results"] if r["backend"] != "exact"}
        if "bpla-dyadic" in rows and "ptq-w8a8" in rows:
            ratio = rows["ptq-w8a8"]["logit_nrmse"] / rows["bpla-dyadic"]["logit_nrmse"]
            print(
                f"  {path.stem}: B-PLA dyadic is {ratio:.0f}x closer than W8A8 per-token "
                f"({rows['bpla-dyadic']['argmax_agreement']:.2f}% vs "
                f"{rows['ptq-w8a8']['argmax_agreement']:.2f}% agreement)"
            )


if __name__ == "__main__":
    main()
