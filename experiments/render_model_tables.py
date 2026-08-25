"""
Render the model-probe JSON records into paper-ready tables.

Reads any number of ``pao_vs_bpla_model.py`` output files and emits both a
Markdown view for reading and a LaTeX ``booktabs`` view for the manuscript, so
the numbers in the paper come from the artifacts rather than being retyped.

Each input file is one matched comparison: a single checkpoint, sample list and
seed, with the exact model as the reference for every approximate row. Files are
never merged, because two runs may differ in evaluation size or replacement
scope and averaging across them would be meaningless.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_LABEL = {
    "exact": "Exact",
    "pao": r"\pam{}",
    "bpla-float": r"\bpla{} float",
    "bpla-dyadic": r"\bpla{} dyadic",
}
BACKEND_PLAIN = {
    "exact": "Exact",
    "pao": "PAM",
    "bpla-float": "B-PLA float",
    "bpla-dyadic": "B-PLA dyadic",
}


def _scope_note(record: dict) -> str:
    config = record["configuration"]
    parts = []
    if config.get("replace_lm_head"):
        parts.append("output projection converted")
    if config.get("replace_conv2d"):
        parts.append("patch embedding converted")
    if config.get("pao_alpha"):
        parts.append(f"PAM alpha={config['pao_alpha']}")
    return "; ".join(parts) if parts else "transformer blocks only"


def _rows(record: dict) -> list[dict]:
    out = []
    for entry in record["results"]:
        task = entry.get("perplexity")
        metric = f"{task:.3f}" if task is not None else f"{entry.get('top1', float('nan')):.2f}"
        out.append(
            {
                "backend": entry["backend"],
                "metric": metric,
                "nrmse": entry.get("logit_nrmse"),
                "agreement": entry.get("argmax_agreement"),
                "gain": entry.get("output_gain"),
                "sites": entry["coverage"].get("linear_modules", 0),
                "seconds": entry.get("emulated_forward_seconds"),
            }
        )
    return out


def _fmt(value: float | None, spec: str, dash: str = "---") -> str:
    return dash if value is None else format(value, spec)


def render(path: Path, latex: bool) -> str:
    record = json.loads(path.read_text(encoding="utf-8"))
    config = record["configuration"]
    model = config["models"][0]
    rows = _rows(record)
    exact = record["results"][0]

    if model == "gpt2":
        scale = f"{exact.get('tokens', 0)} tokens, WikiText-2"
        task_name = "PPL"
    else:
        scale = f"{exact.get('samples', 0)} images, Imagenette"
        task_name = "top-1 (\\%)" if latex else "top-1 (%)"

    caption = (
        f"{'GPT-2' if model == 'gpt2' else 'ViT-Base'}, {scale}. "
        f"Scope: {_scope_note(record)}. "
        f"$k={config['prefix_bits']}$, $T={config['dyadic_terms']}$. "
        "Zero weight updates in every row; logit metrics are row-centered."
    )

    if not latex:
        lines = [f"### {path.stem}", "", caption.replace("$", "").replace("\\%", "%"), ""]
        lines.append(f"| Backend | {task_name} | logit NRMSE | agreement | gain | sites |")
        lines.append("|---|---|---|---|---|---|")
        for row in rows:
            lines.append(
                f"| {BACKEND_PLAIN.get(row['backend'], row['backend'])} "
                f"| {row['metric']} "
                f"| {_fmt(row['nrmse'], '.2e', '—')} "
                f"| {_fmt(row['agreement'], '.2f', '—')} "
                f"| {_fmt(row['gain'], '.4f', '—')} "
                f"| {row['sites']} |"
            )
        return "\n".join(lines)

    lines = [
        r"\begin{table}[t]",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:{path.stem}}}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        rf"Backend & {task_name} & Logit NRMSE & Agreement & Gain & Sites \\",
        r"\midrule",
    ]
    for row in rows:
        agreement = _fmt(row["agreement"], ".2f")
        if agreement != "---":
            agreement += r"\%"
        label = BACKEND_LABEL.get(row["backend"], row["backend"])
        cells = [label, row["metric"], _fmt(row["nrmse"], ".2e"), agreement,
                 _fmt(row["gain"], ".4f"), str(row["sites"])]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    # Tables carry en dashes and LaTeX; a console in a legacy codepage would
    # otherwise fail on them.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Render model-probe JSON into tables.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    for path in args.inputs:
        if not path.exists():
            print(f"(missing: {path})\n")
            continue
        print(render(path, args.latex))
        print()


if __name__ == "__main__":
    main()
