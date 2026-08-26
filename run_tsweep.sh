#!/usr/bin/env bash
# Nonlinear term-budget sweep.
#
# The scope results showed B-PLA holding 99.9% next-token agreement when only
# multiplications are replaced but 91.3% once the nonlinear path is, while its
# float path stayed at 97.3%. That points at dyadic quantization of the
# reciprocal and reciprocal-square-root tables rather than at the tables
# themselves, and predicts that raising only the nonlinear budget closes the
# gap. This sweeps it while holding the multiplier at T=2.
set -u
cd /workspace/B-PLA
source /venv/main/bin/activate

export BPLA_COMPILE=0
export BPLA_MATMUL_ELEMENT_BUDGET=32000000
OUT=results
mkdir -p "$OUT"
LOG="$OUT/driver_tsweep.log"

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

say "=== nonlinear term sweep: multiplier held at T=2 ==="

# Nonlinear scope isolates the nonlinear tables: every matmul stays exact, so
# any change here is attributable to the term budget alone.
for T in 2 3 4 6; do
  run "n1_gpt2_nonlinear_T${T}" \
    --models gpt2 --backends exact bpla-dyadic --scopes nonlinear \
    --gpt2-sequence-length 256 --gpt2-target-tokens "$GPT2_TOKENS" \
    --calibration-batches 2 --replace-layernorm \
    --dyadic-terms 2 --nonlinear-dyadic-terms "$T"

  run "n2_vit_nonlinear_T${T}" \
    --models vit --backends exact bpla-dyadic --scopes nonlinear \
    --num-samples "$VIT_SAMPLES" --batch-size 32 \
    --calibration-batches 2 --replace-layernorm \
    --dyadic-terms 2 --nonlinear-dyadic-terms "$T"
done

# The end state the paper wants: everything replaced, each path at the budget
# it actually needs.
run n3_gpt2_combined_mixed \
  --models gpt2 --backends exact bpla-dyadic --scopes combined \
  --gpt2-sequence-length 256 --gpt2-target-tokens "$GPT2_TOKENS" \
  --calibration-batches 2 --replace-layernorm \
  --dyadic-terms 2 --nonlinear-dyadic-terms 6

run n4_vit_combined_mixed \
  --models vit --backends exact bpla-dyadic --scopes combined \
  --num-samples "$VIT_SAMPLES" --batch-size 32 \
  --calibration-batches 2 --replace-layernorm \
  --dyadic-terms 2 --nonlinear-dyadic-terms 6

say "=== sweep finished ==="
