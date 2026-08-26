#!/usr/bin/env bash
# Overnight GPU queue: the matched PTQ comparison, then the scaling study.
#
# Ordered by what blocks the paper. p1 and p2 are the matched runs the results
# section cannot be written without; everything after them is upside. Each step
# writes a .done marker, so re-running the script resumes rather than repeating
# and an interrupted night costs only the step that was in flight.
#
# Why these backends. ptq-w8a8 is dynamic per-token activation scaling, the
# strong W8A8 recipe; ptq-w8a8-static is the conventional per-tensor one, which
# collapses on GPT-2 and has to be shown rather than hidden. pao and pao-alpha
# are PAM with and without its single-constant correction -- the correction is a
# second piecewise affine multiply, so it doubles PAM's integer additions, and a
# table with only one of the two either understates the baseline or hides that
# cost.
#
# Scope is multiplication throughout: W8A8 leaves GELU, Softmax and LayerNorm in
# floating point by convention, so it has no honest nonlinear or combined row,
# and the harness refuses those combinations rather than mislabelling a run.
set -u
cd /workspace/B-PLA
source /venv/main/bin/activate

export BPLA_COMPILE=0                        # fusion is silently wrong inside
                                             # the model; do not turn this on
export BPLA_MATMUL_ELEMENT_BUDGET=32000000
OUT=results
mkdir -p "$OUT"
LOG="$OUT/driver_ptq.log"

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

ALL="exact ptq-w8a8 ptq-w8a8-static pao pao-alpha bpla-dyadic bpla-float"

say "=== P0: matched W8A8 / PAM / B-PLA comparison ==="

# The two runs the Results section is blocked on. One checkpoint, one sample
# list, one exact reference per run.
run p1_gpt2_ptq \
  --models gpt2 --backends $ALL --scopes multiplication \
  --gpt2-sequence-length 256 --gpt2-target-tokens 25600 \
  --calibration-batches 2 --dyadic-terms 2

run p2_vit_ptq \
  --models vit --backends $ALL --scopes multiplication \
  --num-samples 256 --batch-size 32 \
  --calibration-batches 2 --dyadic-terms 2

say "=== P2: does the gap grow with model scale? ==="

# Activation outliers are known to worsen with model size, which is what makes
# per-tensor W8A8 fail on larger decoders. If B-PLA holds while W8A8 degrades,
# the advantage grows with scale; if B-PLA degrades too, we need to know that
# before claiming otherwise. Token counts shrink as the models grow so each step
# stays within a couple of hours.
run p3_gpt2_medium \
  --models gpt2 --gpt2-model-id gpt2-medium --backends $ALL --scopes multiplication \
  --gpt2-sequence-length 256 --gpt2-target-tokens 12800 \
  --calibration-batches 2 --dyadic-terms 2

run p4_gpt2_large \
  --models gpt2 --gpt2-model-id gpt2-large --backends $ALL --scopes multiplication \
  --gpt2-sequence-length 256 --gpt2-target-tokens 6400 \
  --calibration-batches 2 --dyadic-terms 2

say "=== bonus: wider coverage, only if the night has room ==="

# GPT-2's output projection is 31% of its weighted multiplies and is exact in
# every run above. This is the same comparison with nothing weighted left out.
run p5_gpt2_full_coverage \
  --models gpt2 --backends $ALL --scopes multiplication \
  --gpt2-sequence-length 256 --gpt2-target-tokens 12800 \
  --calibration-batches 2 --dyadic-terms 2 --replace-lm-head

run p6_vit_large \
  --models vit --vit-model-id google/vit-large-patch16-224 --backends $ALL \
  --scopes multiplication --num-samples 128 --batch-size 16 \
  --calibration-batches 2 --dyadic-terms 2

say "=== queue finished ==="
python experiments/render_model_tables.py "$OUT"/p*.json 2>&1 | tee -a "$LOG"
