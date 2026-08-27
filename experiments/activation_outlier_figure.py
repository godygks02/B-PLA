"""
Why per-tensor W8A8 collapses on GPT-2, and why the strong recipe does not.

This figure exists to answer one objection: that the W8A8 baseline was built
badly. The comparison in this paper turns on W8A8's fidelity, so a reader is
entitled to ask whether a better-configured baseline would have closed the gap.
The answer is measured rather than argued, and it is not the answer the usual
advice predicts.

The two panels
--------------
**Left -- the cause.** Per-channel maximum activation at the input of every
transformer block, plotted against channel index so the structure is visible
rather than averaged away. Four channels -- 64, 87, 266 and 480 -- carry the
largest magnitudes in nine or ten of the twelve blocks, peaking around fifteen
times the median. This is the outlier-feature structure LLM.int8() reported,
milder here than in the multi-billion-parameter models that work studied, but
already enough to break a single per-tensor scale: the scale has to span the
outliers, so every other channel gets a fraction of the available levels.

**Right -- the consequence.** Argmax agreement against the exact model for each
recipe. Percentile clipping is the standard prescription for exactly this
problem, and here it makes things *worse*, monotonically: the outlier features
are not noise to be clipped away, they carry signal the model depends on. What
works is giving each token its own scale, which needs no clipping at all.

What this establishes for the comparison
----------------------------------------
Weight quantization is nearly free on GPT-2; essentially the whole loss is in
the activation path. So the W8A8 row in the model tables is not weak because we
quantized weights carelessly, and it is not weak because we picked a bad
percentile -- the best static per-tensor configuration is still far behind
per-token, and per-token is what we report as the primary baseline.

Usage
-----
    python experiments/activation_outlier_figure.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.torch_ptq import (
    TorchPTQConfig,
    calibrate_ptq_model,
    finalize_ptq_model,
    replace_ptq_attention_matmuls,
    replace_ptq_gpt2_conv1d,
)

COLOR_PER_TOKEN = "#1f77b4"
COLOR_PER_TENSOR = "#d62728"
COLOR_WEIGHTS_ONLY = "#7f7f7f"


def build_batches(args) -> list[dict[str, torch.Tensor]]:
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_id)
    raw = load_dataset(args.dataset_id, args.dataset_config, split="test")
    text = "\n\n".join(line for line in raw["text"] if line.strip())
    encoded = tokenizer(text, return_tensors="pt").input_ids[0]
    window = args.sequence_length
    batches = []
    total = 0
    for start in range(0, encoded.numel() - window, window):
        batches.append({"input_ids": encoded[start : start + window].unsqueeze(0)})
        total += window
        if total >= args.target_tokens:
            break
    return batches


def load_model(args):
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(args.model_id).eval()
    return model.to(args.device)


@torch.no_grad()
def measure_block_input_ranges(model, batches, max_batches: int) -> list[list[float]]:
    """Per-channel max |activation| at the input of every transformer block.

    The residual stream is where the outlier features live, and it is what feeds
    the ``c_attn`` projection whose activation scale W8A8 has to choose.
    """

    from transformers.pytorch_utils import Conv1D

    per_layer: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(index: int):
        def hook(_module, inputs):
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return
            amax = x.detach().abs().reshape(-1, x.shape[-1]).amax(dim=0).float().cpu()
            previous = per_layer.get(index)
            per_layer[index] = amax if previous is None else torch.maximum(previous, amax)

        return hook

    index = 0
    for block in model.transformer.h:
        target = block.attn.c_attn
        if isinstance(target, Conv1D):
            hooks.append(target.register_forward_pre_hook(make_hook(index)))
            index += 1

    try:
        for count, batch in enumerate(batches):
            if count >= max_batches:
                break
            model(**batch)
    finally:
        for hook in hooks:
            hook.remove()

    return [per_layer[i].tolist() for i in sorted(per_layer)]


@torch.no_grad()
def evaluate(model, batches) -> tuple[torch.Tensor, float]:
    logits = []
    negative_log_likelihood = 0.0
    tokens = 0
    for batch in batches:
        output = model(**batch).logits.float()
        logits.append(output.reshape(-1, output.size(-1)).cpu())
        loss = nn.functional.cross_entropy(
            output[:, :-1, :].reshape(-1, output.size(-1)),
            batch["input_ids"][:, 1:].reshape(-1),
            reduction="sum",
        )
        negative_log_likelihood += float(loss)
        tokens += int(batch["input_ids"][:, 1:].numel())
    return torch.cat(logits), math.exp(negative_log_likelihood / tokens)


def agreement(current: torch.Tensor, reference: torch.Tensor) -> float:
    return float((current.argmax(-1) == reference.argmax(-1)).float().mean() * 100.0)


def run_recipe(args, batches, config: TorchPTQConfig) -> tuple[torch.Tensor, float]:
    model = load_model(args)
    replace_ptq_gpt2_conv1d(model, config)
    replace_ptq_attention_matmuls(model, config, mode="ptq-full")
    calibrate_ptq_model(model, batches, lambda m, b: m(**b), args.calibration_batches)
    finalize_ptq_model(model)
    outcome = evaluate(model, batches)
    del model
    return outcome


def _recipes(args) -> list[tuple[str, str, TorchPTQConfig]]:
    """The ladder, ordered as the figure reads it."""

    percentiles = [99.9, 99.99, 99.999, 100.0]
    ladder: list[tuple[str, str, TorchPTQConfig]] = [
        (
            "weights only",
            "weights-only",
            # 16-bit activations with no clipping are effectively exact, which
            # isolates the weight path.
            TorchPTQConfig(activation_bits=16, activation_percentile=100.0),
        )
    ]
    for percentile in percentiles:
        label = "per-tensor\nmin-max" if percentile == 100.0 else f"per-tensor\n{percentile}%"
        ladder.append((label, "per-tensor", TorchPTQConfig(activation_percentile=percentile)))
    ladder.append(
        (
            "per-token\ndynamic",
            "per-token",
            TorchPTQConfig(activation_granularity="token"),
        )
    )
    return ladder


def _persistent_channels(ranges: list[list[float]], top: int = 5, minimum: int = 8) -> list[int]:
    """Channels among the largest in at least ``minimum`` of the blocks.

    A channel that is merely large once is noise; one that leads at nearly every
    depth is a feature the model routes through, which is why clipping it costs
    accuracy instead of saving it.
    """

    from collections import Counter

    counter: Counter[int] = Counter()
    for amax in ranges:
        order = sorted(range(len(amax)), key=lambda c: -amax[c])
        counter.update(order[:top])
    return sorted(channel for channel, count in counter.items() if count >= minimum)


def render(record: dict, output: Path) -> Path:
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.4))

    # --- left: the outlier structure -------------------------------------
    ranges = record["block_input_ranges"]
    layers = len(ranges)
    colours = plt.cm.viridis([i / max(1, layers - 1) for i in range(layers)])
    for index, amax in enumerate(ranges):
        # Against channel index, not sorted: sorting each block independently
        # would hide the finding, which is that the *same* channels dominate at
        # every depth.
        left.plot(amax, color=colours[index], linewidth=0.6, alpha=0.55)

    # Twelve overlaid traces read as noise, so the envelope carries the shape and
    # the individual blocks sit behind it as evidence that it is not one layer's
    # accident.
    envelope = [max(amax[c] for amax in ranges) for c in range(len(ranges[0]))]
    left.plot(envelope, color="0.15", linewidth=0.8, zorder=4, label="max over blocks")

    persistent = _persistent_channels(ranges)
    if persistent:
        left.scatter(
            persistent,
            [envelope[c] for c in persistent],
            s=26,
            facecolors="none",
            edgecolors=COLOR_PER_TENSOR,
            linewidths=1.2,
            zorder=5,
            label="leads in $\geq$8 of 12 blocks",
        )
        for order, channel in enumerate(persistent):
            neighbouring = order > 0 and channel - persistent[order - 1] < 40
            left.annotate(
                str(channel),
                (channel, envelope[channel]),
                textcoords="offset points",
                xytext=(6 if neighbouring else -6, 8 if neighbouring else 4),
                ha="left" if neighbouring else "right",
                fontsize=6.5,
                color=COLOR_PER_TENSOR,
            )

    left.set_yscale("log")
    left.set_xlabel("channel index")
    left.set_ylabel(r"max $|x|$ over tokens")
    left.set_title("Block-input activations, per channel", fontsize=9)
    left.grid(alpha=0.3, which="both")

    peak_ratio = max(
        max(amax) / sorted(amax)[len(amax) // 2] for amax in ranges
    )
    left.annotate(
        f"peak / median up to {peak_ratio:.0f}x",
        xy=(0.97, 0.93),
        xycoords="axes fraction",
        ha="right",
        fontsize=7.5,
        color="0.25",
    )
    handles, labels = left.get_legend_handles_labels()
    handles = [
        plt.Line2D([], [], color=colours[0], linewidth=1.4, label="block 0"),
        plt.Line2D([], [], color=colours[-1], linewidth=1.4, label=f"block {layers - 1}"),
    ] + handles
    left.legend(handles=handles, fontsize=6.5, loc="lower left", frameon=False, ncol=2)
    left.set_ylim(top=max(envelope) * 4.0)

    # --- right: what each recipe costs ------------------------------------
    rows = record["recipes"]
    positions = range(len(rows))
    palette = {
        "weights-only": COLOR_WEIGHTS_ONLY,
        "per-tensor": COLOR_PER_TENSOR,
        "per-token": COLOR_PER_TOKEN,
    }
    right.bar(
        list(positions),
        [r["agreement"] for r in rows],
        color=[palette[r["family"]] for r in rows],
        width=0.68,
    )
    for position, row in zip(positions, rows):
        right.annotate(
            f"{row['agreement']:.1f}",
            (position, row["agreement"]),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=6.5,
        )
    right.set_xticks(list(positions))
    right.set_xticklabels(
        [r["label"].replace("\n", " ") for r in rows], fontsize=6.2, rotation=30, ha="right"
    )
    right.set_ylabel("argmax agreement with exact (%)")
    right.set_ylim(0, 108)
    right.set_title("What each W8A8 recipe costs", fontsize=9)
    right.grid(alpha=0.3, axis="y")
    # Clipping is the standard prescription; the arrow says it runs the wrong way.
    clipped = [i for i, r in enumerate(rows) if r["family"] == "per-tensor"]
    if len(clipped) >= 2:
        # Above the bars, not through them: the arrow says the standard
        # prescription runs the wrong way, and it should not fight the data.
        right.annotate(
            "",
            xy=(clipped[0], 84),
            xytext=(clipped[-1], 84),
            arrowprops=dict(arrowstyle="->", color="0.35", linewidth=0.9),
        )
        right.annotate(
            "more clipping, worse agreement",
            xy=((clipped[0] + clipped[-1]) / 2, 86),
            ha="center",
            fontsize=6.5,
            color="0.35",
        )

    figure.suptitle(
        "GPT-2 activation outliers and the W8A8 recipes that do and do not survive them",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the activation-outlier figure.")
    parser.add_argument("--model-id", default="gpt2")
    parser.add_argument("--dataset-id", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--target-tokens", type=int, default=2560)
    parser.add_argument("--calibration-batches", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--data", type=Path, default=Path(__file__).resolve().parent / "activation_outliers.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "fig_activation_outliers.png",
    )
    parser.add_argument(
        "--reuse", action="store_true", help="Render from the existing JSON without remeasuring."
    )
    args = parser.parse_args()

    if args.reuse and args.data.exists():
        record = json.loads(args.data.read_text(encoding="utf-8"))
    else:
        batches = [
            {key: value.to(args.device) for key, value in batch.items()}
            for batch in build_batches(args)
        ]
        exact_model = load_model(args)
        print("measuring block-input ranges ...", flush=True)
        ranges = measure_block_input_ranges(exact_model, batches, args.calibration_batches)
        reference_logits, exact_perplexity = evaluate(exact_model, batches)
        print(f"  exact perplexity {exact_perplexity:.4f}", flush=True)
        del exact_model

        recipes = []
        for label, family, config in _recipes(args):
            logits, perplexity = run_recipe(args, batches, config)
            row = {
                "label": label,
                "family": family,
                "perplexity": perplexity,
                "agreement": agreement(logits, reference_logits),
                "activation_bits": config.activation_bits,
                "activation_percentile": config.activation_percentile,
                "activation_granularity": config.activation_granularity,
            }
            recipes.append(row)
            print(
                f"  {label.replace(chr(10), ' '):22s} ppl={perplexity:9.4f}  "
                f"agree={row['agreement']:7.3f}",
                flush=True,
            )

        record = {
            "configuration": {
                k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
            },
            "exact_perplexity": exact_perplexity,
            "block_input_ranges": ranges,
            "recipes": recipes,
        }
        args.data.write_text(json.dumps(record), encoding="utf-8")

    path = render(record, args.output)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
