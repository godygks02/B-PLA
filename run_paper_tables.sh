#!/usr/bin/env bash
# Paper-table experiment driver. Runs sequentially so the GPU is never shared,
# streams progress to results/driver.log, and writes one JSON per table.
set -u
cd /workspace/B-PLA
source /venv/main/bin/activate

export BPLA_MATMUL_ELEMENT_BUDGET=32000000
CHUNK=512
OUT=results
mkdir -p "$OUT"
LOG="$OUT/driver.log"

GPT2_TOKENS=${GPT2_TOKENS:-25600}
GPT2_HEAD_TOKENS=${GPT2_HEAD_TOKENS:-5120}
VIT_SAMPLES=${VIT_SAMPLES:-512}

say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

run() {
  local name="$1"; shift
  if [ -f "$OUT/$name.done" ]; then say "SKIP $name (already done)"; return; fi
  say "START $name"
  if python -u experiments/pao_vs_bpla_model.py "$@" \
        --linear-chunk-out "$CHUNK" --save-logits \
        --output "$OUT/$name.json" >>"$LOG" 2>&1; then
    touch "$OUT/$name.done"; say "DONE  $name"
  else
    say "FAIL  $name (see $LOG)"
  fi
}

say "=== driver start: gpt2_tokens=$GPT2_TOKENS head_tokens=$GPT2_HEAD_TOKENS vit=$VIT_SAMPLES ==="

# Table 1 - GPT-2, transformer blocks only. The main matched comparison.
run t1_gpt2_blocks \
  --models gpt2 --backends exact pao bpla-float bpla-dyadic --scopes multiplication \
  --gpt2-sequence-length 256 --gpt2-target-tokens "$GPT2_TOKENS" --calibration-batches 2

# Table 2 - the same baseline with the Sec. 2.7 alpha correction fitted, so the
# comparison is against PAM's corrected form rather than its published default.
run t2_gpt2_pao_alpha \
  --models gpt2 --backends exact pao --scopes multiplication \
  --gpt2-sequence-length 256 --gpt2-target-tokens "$GPT2_TOKENS" --calibration-batches 2 \
  --pao-alpha 1.056

# Table 3 - GPT-2 at 100% weighted coverage. Slower: the vocabulary projection
# is heavier than all twelve blocks together, so this runs fewer tokens.
run t3_gpt2_full_coverage \
  --models gpt2 --backends exact pao bpla-dyadic --scopes multiplication \
  --gpt2-sequence-length 256 --gpt2-target-tokens "$GPT2_HEAD_TOKENS" --calibration-batches 1 \
  --replace-lm-head

# Table 4 - ViT, blocks only.
run t4_vit_blocks \
  --models vit --backends exact pao bpla-float bpla-dyadic --scopes multiplication \
  --num-samples "$VIT_SAMPLES" --batch-size 16 --calibration-batches 2

# Table 5 - ViT at 100% weighted coverage (patch embedding converted).
run t5_vit_full_coverage \
  --models vit --backends exact pao bpla-dyadic --scopes multiplication \
  --num-samples "$VIT_SAMPLES" --batch-size 16 --calibration-batches 2 \
  --replace-conv2d

say "=== driver finished ==="
