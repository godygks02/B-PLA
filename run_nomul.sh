#!/usr/bin/env bash
# Re-measure the headline conditions now that no exact multiplication remains
# in either backend: the Softmax log2(e) scaling and the attention scaling now
# go through the multipliers, and B-PLA gained exactness on powers of two plus
# special-value handling.
set -u
cd /workspace/B-PLA
source /venv/main/bin/activate

export BPLA_COMPILE=0
export BPLA_MATMUL_ELEMENT_BUDGET=32000000
OUT=results
mkdir -p "$OUT"
LOG="$OUT/driver_nomul.log"

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

say "=== no exact multiplication remaining, mult T=2 / nonlinear T=4 ==="

# Full weighted coverage plus every nonlinear path: the condition the paper
# reports, now with nothing left exact but the accumulation.
run m1_gpt2_full_nomul \
  --models gpt2 --backends exact pao bpla-float bpla-dyadic --scopes combined \
  --gpt2-sequence-length 256 --gpt2-target-tokens 5120 \
  --calibration-batches 1 --replace-layernorm --replace-lm-head \
  --dyadic-terms 2 --nonlinear-dyadic-terms 4

run m2_vit_full_nomul \
  --models vit --backends exact pao bpla-float bpla-dyadic --scopes combined \
  --num-samples "$VIT_SAMPLES" --batch-size 32 \
  --calibration-batches 2 --replace-layernorm --replace-conv2d \
  --dyadic-terms 2 --nonlinear-dyadic-terms 4

# The multiplication-only scope too, since the attention scaling moved.
run m3_gpt2_mult_nomul \
  --models gpt2 --backends exact pao bpla-float bpla-dyadic --scopes multiplication \
  --gpt2-sequence-length 256 --gpt2-target-tokens "$GPT2_TOKENS" \
  --calibration-batches 2 --dyadic-terms 2

run m4_vit_mult_nomul \
  --models vit --backends exact pao bpla-float bpla-dyadic --scopes multiplication \
  --num-samples "$VIT_SAMPLES" --batch-size 32 \
  --calibration-batches 2 --dyadic-terms 2

say "=== no-exact-multiply pass finished ==="
