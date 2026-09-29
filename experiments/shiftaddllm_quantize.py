"""Reparameterize a pretrained GPT-2 (or OPT) with ShiftAddLLM and save the weights.

    python experiments/shiftaddllm_quantize.py --model gpt2 --mode lat --wbits 3 \
        --seed 0 --output results/shiftaddllm_weights/gpt2_lat3_s0.pt

The quantizer is the authors' code at a pinned commit (see
``modules/torch_shiftaddllm.py``); this script supplies the calibration data,
the block walk and the bookkeeping. The saved checkpoint holds every block
weight matrix after reparameterization, in the layout of the original
checkpoint, and is what the ``shiftaddllm`` backend of
``experiments/pao_vs_bpla_model.py`` loads for the Table 6 row.

``--official-eval`` also scores the model before and after, under the
upstream evaluation protocol. On ``facebook/opt-125m`` that reproduces the
paper's Table 2 and is the check that this port quantizes the way the authors'
scripts do; on GPT-2 it is context only, since Table 6 uses the common
harness's 256-token windows.

Calibration follows upstream: 128 random windows of the model's full context
from the WikiText-2 *training* split. The other training-free backends of
Table 6 calibrate on two 256-token windows of the evaluation split, so this is
more data, from a disjoint split -- the authors' recipe, reported as such.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.torch_shiftaddllm import (  # noqa: E402
    PAPER_OPT125M_WIKITEXT2,
    UPSTREAM_COMMIT,
    binary_code_check,
    conv1d_to_linear,
    cpu_fallback,
    decoder_blocks,
    find_layers,
    import_upstream,
    official_perplexity,
    quantize_model,
    quantized_weights,
    save_checkpoint,
    upstream_args,
    upstream_commit,
    upstream_root,
    wikitext2_calibration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="gpt2", help="Hub id: gpt2, facebook/opt-125m, ...")
    parser.add_argument("--mode", choices=("acc", "lat"), required=True,
                        help="Upstream --acc (incoherence processing, accuracy-oriented) or "
                             "--lat (block-wise binary codes the LUT kernel runs).")
    parser.add_argument("--wbits", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument("--seed", type=int, default=0,
                        help="Seeds the calibration windows and the --acc random rotations.")
    parser.add_argument("--nsamples", type=int, default=128, help="Calibration windows (upstream default).")
    parser.add_argument("--seqlen", type=int, default=None,
                        help="Calibration window length. Default: the model's full context "
                             "(1024 for GPT-2, 2048 for OPT), as upstream.")
    parser.add_argument("--calibration-batch-size", type=int, default=8)
    parser.add_argument("--bcq-round", type=int, default=50, help="Upstream eval_opt.sh uses 50.")
    parser.add_argument("--apot-nums", type=int, default=3, help="Power-of-two terms per scale (upstream default).")
    parser.add_argument("--groupsize", type=int, default=-1)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--dtype", choices=("auto", "float32", "float16"), default="auto",
                        help="auto keeps the checkpoint dtype, as upstream: float32 for GPT-2, "
                             "float16 for OPT.")
    parser.add_argument("--dataset-id", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--official-eval", action="store_true",
                        help="Score before and after under the upstream opt_eval protocol.")
    parser.add_argument("--official-max-windows", type=int, default=None,
                        help="Smoke tests only: cap the official evaluation.")
    parser.add_argument("--max-blocks", type=int, default=None,
                        help="Smoke tests only: quantize the first N blocks. The checkpoint is "
                             "marked partial and the harness refuses it.")
    parser.add_argument("--upstream", default=None, help="ShiftAddLLM clone (default third_party/ShiftAddLLM).")
    parser.add_argument("--allow-other-commit", action="store_true",
                        help=f"Run even if the clone is not at {UPSTREAM_COMMIT[:12]}.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True, help="Checkpoint (.pt) to write.")
    parser.add_argument("--report", type=Path, default=None,
                        help="JSON summary to write. Default: the checkpoint path with .json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.report or args.output.with_suffix(".json")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = upstream_root(args.upstream)
    commit = upstream_commit(root)
    if commit != UPSTREAM_COMMIT and not args.allow_other_commit:
        raise SystemExit(
            f"{root} is at {commit or 'an unknown commit'}, not {UPSTREAM_COMMIT}. "
            "Run ./fetch_shiftaddllm.sh, or pass --allow-other-commit."
        )
    shiftaddllm_cls, bcquantizer_cls = import_upstream(root)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"auto": "auto", "float32": torch.float32, "float16": torch.float16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.eval().to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    seqlen = args.seqlen or int(model.config.max_position_embeddings)
    device = torch.device(args.device)

    record: dict[str, object] = {
        "model_id": args.model,
        "mode": args.mode,
        "wbits": args.wbits,
        "seed": args.seed,
        "nsamples": args.nsamples,
        "seqlen": seqlen,
        "calibration": f"{args.dataset_id}/{args.dataset_config} train, random windows",
        "bcq_round": args.bcq_round,
        "apot_nums": args.apot_nums,
        "groupsize": args.groupsize,
        "blocksize": args.blocksize,
        "percdamp": args.percdamp,
        "dtype": str(next(model.parameters()).dtype),
        "upstream_commit": commit,
        "device": args.device,
    }

    if args.official_eval:
        print("official-protocol perplexity, before ...", flush=True)
        record["official_before"] = official_perplexity(
            model, tokenizer, seqlen, device, args.dataset_id, args.dataset_config,
            args.official_max_windows,
        )
        print(f"    {record['official_before']}", flush=True)

    conv1d_names = conv1d_to_linear(model)
    record["conv1d_converted"] = len(conv1d_names)

    calibration = wikitext2_calibration(
        tokenizer, args.nsamples, args.seed, seqlen, args.dataset_id, args.dataset_config
    )
    qargs = upstream_args(
        args.mode,
        args.wbits,
        groupsize=args.groupsize,
        blocksize=args.blocksize,
        bcq_round=args.bcq_round,
        apot_nums=args.apot_nums,
        percdamp=args.percdamp,
    )
    record["upstream_args"] = {k: v for k, v in vars(qargs).items()}

    started = time.perf_counter()
    with cpu_fallback() as cpu_smoke:
        blocks = quantize_model(
            model,
            calibration,
            qargs,
            shiftaddllm_cls,
            bcquantizer_cls,
            model_name=args.model.split("/")[-1],
            batch_size=args.calibration_batch_size,
            max_blocks=args.max_blocks,
        )
    record["quantize_seconds"] = time.perf_counter() - started
    record["blocks"] = blocks
    total_blocks = len(decoder_blocks(model))
    record["blocks_quantized"] = len(blocks)
    record["partial"] = len(blocks) < total_blocks
    record["cpu_smoke"] = bool(cpu_smoke)

    # Is what we are about to evaluate a binary code, or a rotated dense matrix?
    checks = {}
    for index, block in enumerate(decoder_blocks(model)[: len(blocks)]):
        for name, module in find_layers(block).items():
            checks[f"{index}.{name}"] = binary_code_check(module.weight.data, args.wbits)
    record["binary_code"] = {
        "all_conclusive": all(c["conclusive"] for c in checks.values()),
        "all_binary_coded": all(c["binary_coded"] for c in checks.values()),
        "max_distinct_per_group": max(c["max_distinct_per_group"] for c in checks.values()),
        "limit": 2 ** args.wbits,
        "matrices": len(checks),
        "per_matrix": checks,
    }
    print(
        f"binary-coded: {record['binary_code']['all_binary_coded']} "
        f"(max {record['binary_code']['max_distinct_per_group']} distinct values per "
        f"column group, limit {2 ** args.wbits})",
        flush=True,
    )

    if args.official_eval:
        print("official-protocol perplexity, after ...", flush=True)
        record["official_after"] = official_perplexity(
            model, tokenizer, seqlen, device, args.dataset_id, args.dataset_config,
            args.official_max_windows,
        )
        print(f"    {record['official_after']}", flush=True)
        if args.model == "facebook/opt-125m":
            record["paper_opt125m"] = {
                "fp16": PAPER_OPT125M_WIKITEXT2["fp16"],
                "quantized": PAPER_OPT125M_WIKITEXT2.get((args.mode, args.wbits)),
            }

    weights = quantized_weights(model, conv1d_names, blocks=len(blocks))
    meta = {key: value for key, value in record.items() if key not in {"blocks", "binary_code"}}
    meta["binary_coded"] = record["binary_code"]["all_binary_coded"]
    save_checkpoint(args.output, weights, meta)
    record["checkpoint"] = str(args.output)
    record["weights_saved"] = len(weights)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {args.output} ({len(weights)} matrices) and {report_path}", flush=True)


if __name__ == "__main__":
    main()
