#!/usr/bin/env python3
"""Summarize the ShiftAddLLM runs of run_vast_shiftaddllm.sh.

    python report_shiftaddllm.py                         # reads results/
    python report_shiftaddllm.py --results experiments/vast_results --latex

Prints three things:

1. The port check: OPT-125M WikiText-2 perplexity under the authors' protocol
   next to the paper's Table 2. If these disagree, the GPT-2 rows are not
   ShiftAddLLM and must not be reported.
2. The GPT-2 rows in the Table 6 setting, mean and sample standard deviation
   over the quantization seeds, with whether the evaluated weights are really
   binary codes (Lat.) or rotated dense matrices (Acc.).
3. The existing Table 6 GPT-2 weighted rows for context, with their own exact
   reference, because a different GPU can give a different exact perplexity
   and rows with different references do not belong in one column.

Smoke-test artifacts (CPU fallback, partial checkpoints) are refused.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

CONFIG_LABELS = {
    "acc3": "ShiftAddLLM (Acc.) 3-bit",
    "acc2": "ShiftAddLLM (Acc.) 2-bit",
    "lat3": "ShiftAddLLM (Lat.) 3-bit",
    "lat2": "ShiftAddLLM (Lat.) 2-bit",
}
PAPER_OPT125M = {"fp16": 27.65, "acc3": 31.29, "acc2": 51.15, "lat3": 56.96, "lat2": 712.55}

#: Table 6 GPT-2 weighted rows already measured on the H100, for context.
CONTEXT = (
    ("experiments/h100_results/table5_gpt2_full.json", "bpla-dyadic", "B-PLA SPT (k=4, T=2)"),
    ("experiments/h100_results/t6_chenpam_gpt2.json", "chen-pam", "Chen-PAM (k=4)"),
    ("experiments/h100_results/t6_smoothquant_gpt2.json", "ptq-smoothquant", "SmoothQuant W8A8"),
    ("experiments/h100_results/table5_gpt2_full.json", "pao", "PAO (FWD-only)"),
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_backend(result: dict) -> dict[str, dict]:
    return {row["backend"]: row for row in result["results"]}


def spread(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    return mean, (statistics.stdev(values) if len(values) > 1 else float("nan"))


def fmt(mean: float, std: float, digits: int, percent: bool = False, sci: bool = False) -> str:
    unit = "%" if percent else ""
    body = f"{mean:.{digits}e}" if sci else f"{mean:.{digits}f}"
    if math.isnan(std):
        return f"{body}{unit}"
    tail = f"{std:.{max(digits - 1, 1)}e}" if sci else f"{std:.{digits}f}"
    return f"{body} ± {tail}{unit}"


def refuse_smoke(meta: dict, where: Path) -> str | None:
    if meta.get("cpu_smoke"):
        return f"{where}: CPU smoke run"
    if meta.get("partial"):
        return f"{where}: only {meta.get('blocks_quantized')} blocks quantized"
    return None


def port_check(weights_dir: Path) -> list[str]:
    lines = ["## 1. Port check: OPT-125M, authors' protocol", "",
             "| config | FP16 ours | FP16 paper | quantized ours | quantized paper | binary-coded |",
             "|---|---:|---:|---:|---:|---|"]
    found = False
    for path in sorted(weights_dir.glob("opt125m_*_s*.json")):
        if path.name.endswith(".partial.json"):
            continue
        report = load(path)
        problem = refuse_smoke(report, path)
        if problem:
            lines.append(f"| refused | {problem} | | | | |")
            continue
        config = f"{report['mode']}{report['wbits']}"
        before = report.get("official_before", {}).get("perplexity", float("nan"))
        after = report.get("official_after", {}).get("perplexity", float("nan"))
        lines.append(
            f"| {config} s{report['seed']} | {before:.2f} | {PAPER_OPT125M['fp16']:.2f} | "
            f"{after:.2f} | {PAPER_OPT125M.get(config, float('nan')):.2f} | "
            f"{report['binary_code']['all_binary_coded']} |"
        )
        found = True
    if not found:
        lines.append("| (no OPT-125M reports yet) | | | | | |")
    return lines


def gpt2_rows(results_dir: Path, weights_dir: Path, latex: bool) -> tuple[list[str], float | None]:
    by_config: dict[str, list[dict]] = {}
    exact_ppl: set[float] = set()
    refused: list[str] = []
    for path in sorted(results_dir.glob("shiftaddllm_gpt2_*_s*.json")):
        if path.name.endswith(".partial"):
            continue
        result = load(path)
        rows = rows_by_backend(result)
        row = rows.get("shiftaddllm")
        if row is None:
            continue
        meta = row["coverage"].get("shiftaddllm", {})
        problem = refuse_smoke(meta, path)
        if problem:
            refused.append(problem)
            continue
        config = f"{meta['mode']}{meta['wbits']}"
        quantize_report = weights_dir / f"gpt2_{config}_s{meta['seed']}.json"
        binary = None
        if quantize_report.is_file():
            binary = load(quantize_report)["binary_code"]["all_binary_coded"]
        exact_ppl.add(round(rows["exact"]["perplexity"], 6))
        by_config.setdefault(config, []).append({**row, "binary": binary, "seed": meta["seed"]})

    lines = ["## 2. GPT-2, Table 6 setting (WikiText-2 test, 256-token windows)", ""]
    reference = None
    if len(exact_ppl) > 1:
        lines.append(f"**WARNING: runs disagree on the exact reference: {sorted(exact_ppl)}**")
    elif exact_ppl:
        reference = next(iter(exact_ppl))
        lines.append(f"Exact reference PPL {reference:.4f} (shared by every run).")
    lines += ["", "| method | seeds | PPL | logit NRMSE | agreement | binary-coded weights |",
              "|---|---:|---:|---:|---:|---|"]
    latex_rows = []
    for config in ("acc3", "acc2", "lat3", "lat2"):
        runs = by_config.get(config)
        if not runs:
            continue
        ppl = spread([r["perplexity"] for r in runs])
        nrmse = spread([r["logit_nrmse"] for r in runs])
        agree = spread([r["argmax_agreement"] for r in runs])
        binary = {r["binary"] for r in runs}
        tokens = {r["tokens"] for r in runs}
        lines.append(
            f"| {CONFIG_LABELS[config]} | {len(runs)} ({','.join(str(r['seed']) for r in runs)}) | "
            f"{fmt(*ppl, 3)} | {fmt(*nrmse, 2, sci=True)} | {fmt(*agree, 2, percent=True)} | "
            f"{'yes' if binary == {True} else 'no' if binary == {False} else binary} |"
        )
        if tokens != {284835}:
            lines.append(f"| ^ token count {sorted(tokens)} is not the full split (284,835) | | | | | |")
        mantissa, exponent = f"{nrmse[0]:.2e}".split("e")
        label = CONFIG_LABELS[config].replace("ShiftAddLLM", r"ShiftAddLLM$^{\ddagger}$")
        nrmse_tex = mantissa + r"{\times}10^{" + str(int(exponent)) + "}"
        latex_rows.append(
            f" & {label} & {ppl[0]:.4f} & ${nrmse_tex}$ & ${agree[0]:.2f}" + r"\%$ \\"
        )
    if refused:
        lines += ["", "Refused: " + "; ".join(refused)]
    if latex and latex_rows:
        lines += ["", "```latex", *latex_rows, "```",
                  "% ‡ ShiftAddLLM reparameterizes the block weights into binary codes (Acc.: in a",
                  "% rotated basis, evaluated after rotating back); QK^T, PV and the vocabulary",
                  "% projection stay in floating point. Mean over quantization seeds."]
    return lines, reference


def context_rows(root: Path, reference: float | None) -> list[str]:
    lines = ["## 3. Existing Table 6 GPT-2 weighted rows (H100) for context", "",
             "| method | exact PPL of that run | PPL | logit NRMSE | agreement |",
             "|---|---:|---:|---:|---:|"]
    for relative, backend, label in CONTEXT:
        path = root / relative
        if not path.is_file():
            continue
        rows = rows_by_backend(load(path))
        if backend not in rows:
            continue
        row, exact = rows[backend], rows["exact"]["perplexity"]
        lines.append(
            f"| {label} | {exact:.4f} | {row['perplexity']:.4f} | {row['logit_nrmse']:.2e} | "
            f"{row['argmax_agreement']:.2f}% |"
        )
        if reference is not None and abs(exact - reference) > 1e-4:
            lines.append(f"| ^ exact reference differs from the ShiftAddLLM runs ({reference:.4f}); "
                         "compare within a run, not across machines | | | | |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--latex", action="store_true", help="Also print Table 6 rows in LaTeX.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    results = args.results if args.results.is_absolute() else root / args.results
    weights = results / "shiftaddllm_weights"
    out = port_check(weights)
    gpt2, reference = gpt2_rows(results, weights, args.latex)
    out += [""] + gpt2 + [""] + context_rows(root, reference)
    print("\n".join(out))


if __name__ == "__main__":
    main()
