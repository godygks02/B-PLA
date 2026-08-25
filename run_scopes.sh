#!/usr/bin/env bash
# Operator-sensitivity pass: the nonlinear and combined replacement scopes,
# which the first pass did not cover.
#
# Kernel fusion is OFF. It looked like a 3.2-3.7x speedup and matched eager on
# a single fixed shape, but inside a model it silently produces wrong results
# -- see tests/test_bpla_table_forms.py::CompiledPathTests. Do not re-enable
# without that test passing on this machine.
#
# The full ViT split is not run here: without fusion it is roughly 21 GPU hours
# for four backends, which is a cost decision rather than a default. Set
# RUN_VIT_FULL=1 to include it.
set -u
cd /workspace/B-PLA
source /venv/main/bin/activate

export BPLA_COMPILE=0
export BPLA_MATMUL_ELEMENT_BUDGET=32000000
CHUNK=512
OUT=results
mkdir -p "$OUT"
LOG="$OUT/driver_scopes.log"

GPT2_TOKENS=${GPT2_TOKENS:-25600}
VIT_SCOPE=${VIT_SCOPE:-256}
VIT_FULL=${VIT_FULL:-4000}

say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

run() {
  local name="$1"; shift
  if [ -f "$OUT/$name.done" ]; then say "SKIP $name"; return; fi
  say "START $name"
  local t0=$SECONDS
  if python -u experiments/pao_vs_bpla_model.py "$@" \
        --linear-chunk-out "$CHUNK" --output "$OUT/$name.json" >>"$LOG" 2>&1; then
    touch "$OUT/$name.done"; say "DONE  $name ($(( (SECONDS-t0)/60 )) min)"
  else
    say "FAIL  $name (see $LOG)"
  fi
}

say "=== scope pass: gpt2=$GPT2_TOKENS vit_scope=$VIT_SCOPE fusion=off ==="

# Nonlinear first: it leaves every matmul exact and replaces GELU, Softmax and
# LayerNorm, so it is both the cheapest run and the one isolating the nonlinear
# path. It is also the first model-level check of the reciprocal and rsqrt
# tables at T=2, which only became usable after the expansion-point change.
run s1_gpt2_nonlinear \
  --models gpt2 --backends exact pao bpla-float bpla-dyadic --scopes nonlinear \
  --gpt2-sequence-length 256 --gpt2-target-tokens "$GPT2_TOKENS" --calibration-batches 2 \
  --replace-layernorm

run s2_vit_nonlinear \
  --models vit --backends exact pao bpla-float bpla-dyadic --scopes nonlinear \
  --num-samples "$VIT_SCOPE" --batch-size 32 --calibration-batches 2 \
  --replace-layernorm

run s3_gpt2_combined \
  --models gpt2 --backends exact pao bpla-float bpla-dyadic --scopes combined \
  --gpt2-sequence-length 256 --gpt2-target-tokens "$GPT2_TOKENS" --calibration-batches 2 \
  --replace-layernorm

run s4_vit_combined \
  --models vit --backends exact pao bpla-float bpla-dyadic --scopes combined \
  --num-samples "$VIT_SCOPE" --batch-size 32 --calibration-batches 2 \
  --replace-layernorm

if [ "${RUN_VIT_FULL:-0}" = "1" ]; then
  run s5_vit_full_multiplication \
    --models vit --backends exact pao bpla-float bpla-dyadic --scopes multiplication \
    --num-samples "$VIT_FULL" --batch-size 32 --calibration-batches 2
fi

say "=== scope pass finished ==="
