#!/usr/bin/env bash
# Combined-scope queue: the one comparison where a training-free baseline
# reaches the nonlinear operators.
#
# Standard W8A8 has no combined-scope row -- it leaves GELU, Softmax and
# LayerNorm in floating point -- so at this scope the only training-free
# baseline that exists is FQ-ViT (Lin et al., IJCAI 2022), which quantizes
# LayerNorm with Power-of-Two Factor and Softmax with 4-bit Log-Int-Softmax.
# I-BERT and I-ViT go further but both require quantization-aware fine-tuning,
# so they are not comparable in this setting.
#
# Coverage is not equal and the table must say so: FQ-ViT never addresses GELU,
# so its rows convert 73 linear + 12 attention + 25 LayerNorm sites with the
# activation left exact, while B-PLA converts the activation too. The JSON
# records this as activation_modules=0, which is the difference itself.
#
# Our FQ-ViT is an upper bound on the real thing: PTF and LIS are reproduced
# from the released code, but their integer LayerNorm arithmetic is not, so we
# omit an error source that the real implementation has. A baseline we
# implemented ourselves should err in its favour.
set -u
cd /workspace/B-PLA
source /venv/main/bin/activate

export BPLA_COMPILE=0
export BPLA_MATMUL_ELEMENT_BUDGET=32000000
OUT=results
mkdir -p "$OUT"
LOG="$OUT/driver_fqvit.log"

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

say "=== combined scope: FQ-ViT vs PAM vs B-PLA ==="

# ViT first: this is FQ-ViT's own domain, so it is the row that carries weight.
# T=4 on the nonlinear tables is the operating point the term sweep settled on.
run f1_vit_combined \
  --models vit --backends exact ptq-fqvit pao bpla-dyadic --scopes combined \
  --num-samples 256 --batch-size 32 \
  --calibration-batches 2 --replace-layernorm --replace-conv2d \
  --dyadic-terms 2 --nonlinear-dyadic-terms 4

# GPT-2 second, and it needs a caveat in the table: FQ-ViT is a vision method,
# and applying PTF and LIS to a decoder is our extension of it, not a published
# result. It is worth measuring because it answers whether the mechanisms
# transfer -- a decoder's LayerNorm inputs carry the outlier features that broke
# per-tensor W8A8, which is exactly what PTF is meant to absorb.
run f2_gpt2_combined \
  --models gpt2 --backends exact ptq-fqvit pao bpla-dyadic --scopes combined \
  --gpt2-sequence-length 256 --gpt2-target-tokens 5120 \
  --calibration-batches 1 --replace-layernorm --replace-lm-head \
  --dyadic-terms 2 --nonlinear-dyadic-terms 4

# The Log-Int-Softmax bit width is FQ-ViT's most aggressive choice: 4 bits on
# the attention map. If that is what costs it accuracy rather than PTF, an
# 8-bit variant separates the two, and the comparison should not rest on their
# most aggressive setting alone.
run f3_vit_combined_lis8 \
  --models vit --backends exact ptq-fqvit --scopes combined \
  --num-samples 256 --batch-size 32 --fqvit-softmax-bits 8 \
  --calibration-batches 2 --replace-layernorm --replace-conv2d

say "=== combined-scope queue finished ==="
python experiments/render_model_tables.py "$OUT"/f*.json 2>&1 | tee -a "$LOG"
