#!/usr/bin/env bash
# The recommended operating point: multiplier at T=2, nonlinear tables at T=4.
#
# The sweep put the nonlinear saturation at T=4 -- it reaches the float path's
# error and agreement, and T=6 adds nothing but a third more shift-adds. These
# runs measure that configuration directly so the paper reports what it
# recommends rather than an upper bound.
#
# Waits for any running sweep so the two never share the GPU.
set -u
cd /workspace/B-PLA
source /venv/main/bin/activate

while tmux has-session -t tsweep 2>/dev/null; do sleep 30; done

export BPLA_COMPILE=0
export BPLA_MATMUL_ELEMENT_BUDGET=32000000
OUT=results
mkdir -p "$OUT"
LOG="$OUT/driver_t4.log"

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

say "=== recommended operating point: multiplier T=2, nonlinear T=4 ==="

run r1_gpt2_combined_T4 \
  --models gpt2 --backends exact bpla-dyadic --scopes combined \
  --gpt2-sequence-length 256 --gpt2-target-tokens "$GPT2_TOKENS" \
  --calibration-batches 2 --replace-layernorm \
  --dyadic-terms 2 --nonlinear-dyadic-terms 4

run r2_vit_combined_T4 \
  --models vit --backends exact bpla-dyadic --scopes combined \
  --num-samples "$VIT_SAMPLES" --batch-size 32 \
  --calibration-batches 2 --replace-layernorm \
  --dyadic-terms 2 --nonlinear-dyadic-terms 4

# Full weighted coverage at the same budgets: every multiply converted,
# including GPT-2's output projection.
run r3_gpt2_full_coverage_T4 \
  --models gpt2 --backends exact pao bpla-dyadic --scopes combined \
  --gpt2-sequence-length 256 --gpt2-target-tokens 5120 \
  --calibration-batches 1 --replace-layernorm --replace-lm-head \
  --dyadic-terms 2 --nonlinear-dyadic-terms 4

run r4_vit_full_coverage_T4 \
  --models vit --backends exact pao bpla-dyadic --scopes combined \
  --num-samples "$VIT_SAMPLES" --batch-size 32 \
  --calibration-batches 2 --replace-layernorm --replace-conv2d \
  --dyadic-terms 2 --nonlinear-dyadic-terms 4

say "=== T=4 pass finished ==="
