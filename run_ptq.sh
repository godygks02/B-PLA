#!/usr/bin/env bash
# W8A8 PTQ baseline against PAO and B-PLA, matched run.
#
# Post-training quantization is the standard a training-free method is actually
# measured against, so this is the comparison the paper needs. Every row here
# comes from one checkpoint, one sample list, one seed and one exact reference.
#
# Two W8A8 recipes are reported. ptq-w8a8 uses dynamic per-token activation
# scales (ZeroQuant / LLM.int8() style), which is the strong form; on GPT-2 the
# conventional static per-tensor form collapses, because percentile clipping
# removes exactly the outlier features that model depends on. Reporting only
# one of the two would misrepresent the baseline in one direction or the other.
#
# Scope is multiplication only: W8A8 leaves GELU, Softmax and LayerNorm in
# floating point by convention, so it has no honest nonlinear or combined row.
# The harness refuses those combinations rather than mislabelling a run.
set -u
cd /workspace/B-PLA
source /venv/main/bin/activate

export BPLA_COMPILE=0                        # see the handoff note: fusion is
                                             # silently wrong inside the model
export BPLA_MATMUL_ELEMENT_BUDGET=32000000
OUT=results
mkdir -p "$OUT"
LOG="$OUT/driver_ptq.log"

GPT2_TOKENS=${GPT2_TOKENS:-25600}
VIT_SAMPLES=${VIT_SAMPLES:-256}

say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

run() {
  local name="$1"; shift
  if [ -f "$OUT/$name.done" ]; then say "SKIP $name"; return; fi
  say "START $name"
  local t0=$SECONDS
  if python -u experiments/pao_vs_bpla_model.py "$@" \
        --linear-chunk-out 512 --output "$OUT/$name.json" >>"$LOG" 2>&1; then
    touch "$OUT/$name.done"; say "DONE  $name ($(( SECONDS-t0 ))s)"
  else
    say "FAIL  $name (see $LOG)"
  fi
}

say "=== W8A8 PTQ vs PAM vs B-PLA, multiplication scope ==="

run p1_gpt2_ptq \
  --models gpt2 \
  --backends exact ptq-w8a8 ptq-w8a8-static pao bpla-dyadic bpla-float \
  --scopes multiplication \
  --gpt2-sequence-length 256 --gpt2-target-tokens "$GPT2_TOKENS" \
  --calibration-batches 2 --dyadic-terms 2

run p2_vit_ptq \
  --models vit \
  --backends exact ptq-w8a8 ptq-w8a8-static pao bpla-dyadic bpla-float \
  --scopes multiplication \
  --num-samples "$VIT_SAMPLES" --batch-size 32 \
  --calibration-batches 2 --dyadic-terms 2

say "=== PTQ pass finished ==="
