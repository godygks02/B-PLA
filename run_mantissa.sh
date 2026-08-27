#!/usr/bin/env bash
# Model-level mantissa width sweep: is B-PLA's energy gap against int8 real?
#
# The cost model says the multiplier spends 92% of its energy at T=2 on 3T+2
# fixed-point additions, and their cost is linear in the mantissa datapath
# width. That width is 24 bits only because float32's significand is; nothing
# about the method requires it. The primitive sweep found the knee well below
# 24 -- 16 bits is nearly free and 12 costs about 11% more error -- which would
# move B-PLA from 3.28x int8 energy to 1.78x.
#
# This asks whether that holds on a real model, where errors compound across 48
# matmuls and a residual stream, rather than on sampled operand pairs.
#
# What narrowing gives up, and why the table has to say so: at full width B-PLA
# carries the operand mantissas exactly and multiplies powers of two exactly.
# Below it, both properties go. The residual stays bounded by the datapath
# resolution, so the trade is predictable, but it is a trade.
set -u
cd /workspace/B-PLA
source /venv/main/bin/activate

export BPLA_COMPILE=0
export BPLA_MATMUL_ELEMENT_BUDGET=32000000
OUT=results
mkdir -p "$OUT"
LOG="$OUT/driver_mantissa.log"

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

say "=== mantissa width sweep, multiplication scope ==="

# Widths chosen around the primitive-level knee. int8 and the full-width B-PLA
# rows come along in every run so each width is read against both ends of the
# comparison under one exact reference.
for BITS in 24 16 12 10 8; do
  run "w1_gpt2_mantissa_${BITS}" \
    --models gpt2 --backends exact ptq-w8a8 bpla-dyadic --scopes multiplication \
    --gpt2-sequence-length 256 --gpt2-target-tokens 12800 \
    --calibration-batches 2 --dyadic-terms 2 --mantissa-bits "$BITS"
done

for BITS in 24 16 12 10 8; do
  run "w2_vit_mantissa_${BITS}" \
    --models vit --backends exact ptq-w8a8 bpla-dyadic --scopes multiplication \
    --num-samples 128 --batch-size 32 \
    --calibration-batches 2 --dyadic-terms 2 --mantissa-bits "$BITS"
done

say "=== mantissa sweep finished ==="
python experiments/render_model_tables.py "$OUT"/w*.json 2>&1 | tee -a "$LOG"
