#!/usr/bin/env bash
# ShiftAddLLM row of the Table 6 GPT-2 column, on a vast.ai box.
#
#   ./run_vast_shiftaddllm.sh                           # one worker on GPU 0
#   BPLA_GPUS=0,0,0,0,0,0,0 ./run_vast_shiftaddllm.sh   # seven workers sharing GPU 0
#   BPLA_GPUS=0,1,2,3 ./run_vast_shiftaddllm.sh         # one worker per GPU
#   SAL_CONFIGS="lat3 acc3" SAL_SEEDS=0 ./run_vast_shiftaddllm.sh   # a subset
#   SAL_DRY_RUN=1 ./run_vast_shiftaddllm.sh             # print the plan only
#
# What it runs, in this order:
#
#   1. OPT-125M, seed 0, Acc. 3-bit and Lat. 3-bit, scored with the authors'
#      own protocol. This checks the port against the paper's Table 2
#      (FP16 27.65, Acc. 3-bit 31.29, Lat. 3-bit 56.96) before any GPT-2
#      number is trusted. OPT-125M is a model the authors' scripts support.
#   2. GPT-2, {acc3, acc2, lat3, lat2} x seeds {0, 1, 2}: quantize with the
#      authors' quantizer, then score in the Table 6 GPT-2 setting -- the
#      whole WikiText-2 test split in 256-token windows (284,835 tokens),
#      --stream-metrics, harness seed 0 -- against the exact model, which
#      every job re-runs so it carries its own reference.
#
# The seed moves the 128 calibration windows and, for Acc., the random
# rotations; three seeds give a spread instead of one draw.
#
# Nothing here is emulated: the scored model is an ordinary floating-point
# GPT-2 with replaced weights, about a minute per job on the GPU. The time is
# in the authors' quantizer, which refines one column at a time with
# thousands of tiny operations. That is launch-bound on a GPU, so it runs on
# the CPU (--quantizer-device cpu: 72 ms per 768-row column on one thread
# against 270-370 ms on an RTX 4060 Ti, bit-identical) while calibration stays
# on the GPU. A worker needs about two CPU threads and 1.5 GB of GPU memory,
# so several can share one card: list the same index several times in
# BPLA_GPUS. On the 48-core, 16 GB RTX 4060 Ti box, seven workers get one
# Acc. and one Lat. job each.
#
# Outputs (results/ is copied back afterwards):
#   results/shiftaddllm_weights/<model>_<cfg>_s<seed>.{pt,json}  checkpoint + quantize report
#   results/shiftaddllm_gpt2_<cfg>_s<seed>.json                  Table 6 harness result
#   results/shiftaddllm_*.log, results/vast_shiftaddllm.log      logs
#
# Then:  python report_shiftaddllm.py

set -u

cd "$(dirname "$0")"

# The vast.ai image keeps its torch stack in a venv and leaves no `python` on
# the PATH of a non-interactive shell.
if ! command -v python >/dev/null 2>&1 && [ -f /venv/main/bin/activate ]; then
  # shellcheck disable=SC1091
  . /venv/main/bin/activate
fi

IFS=',' read -r -a GPUS <<<"${BPLA_GPUS:-0}"
read -r -a CONFIGS <<<"${SAL_CONFIGS:-acc3 acc2 lat3 lat2}"
read -r -a SEEDS <<<"${SAL_SEEDS:-0 1 2}"
DRY="${SAL_DRY_RUN:-0}"

OUT=results
WEIGHTS="$OUT/shiftaddllm_weights"
mkdir -p "$OUT" "$WEIGHTS"
LOG="$OUT/vast_shiftaddllm.log"

say() { echo "[$(date -u +'%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# acc3 -> "acc 3"
split_config() { echo "${1%?} ${1: -1}"; }

# Quantize one model/config/seed into $WEIGHTS. Skipped when present.
quantize() {
  local gpu="$1" model="$2" short="$3" config="$4" seed="$5"
  local mode bits
  read -r mode bits <<<"$(split_config "$config")"
  local name="${short}_${config}_s${seed}"
  local log="$OUT/shiftaddllm_quantize_${name}.log"
  if [ -s "$WEIGHTS/$name.pt" ] && [ -s "$WEIGHTS/$name.json" ]; then
    say "SKIP  gpu$gpu quantize $name (already present)"
    return 0
  fi
  say "START gpu$gpu quantize $name"
  [ "$DRY" = 1 ] && return 0
  if CUDA_VISIBLE_DEVICES="$gpu" python experiments/shiftaddllm_quantize.py \
        --model "$model" --mode "$mode" --wbits "$bits" --seed "$seed" \
        --official-eval --device cuda --quantizer-device cpu --cpu-threads 2 \
        --output "$WEIGHTS/$name.partial.pt" --report "$WEIGHTS/$name.partial.json" \
        >"$log" 2>&1; then
    mv "$WEIGHTS/$name.partial.pt" "$WEIGHTS/$name.pt"
    mv "$WEIGHTS/$name.partial.json" "$WEIGHTS/$name.json"
    say "DONE  gpu$gpu quantize $name ($(grep -o "'perplexity': [0-9.]*" "$log" | tail -1))"
  else
    say "FAIL  gpu$gpu quantize $name -- tail of $log:"
    tail -6 "$log" | sed 's/^/    /'
    return 1
  fi
}

# Score one GPT-2 checkpoint in the Table 6 setting. Skipped when present.
evaluate() {
  local gpu="$1" name="$2"
  local target="$OUT/shiftaddllm_${name}.json"
  local log="$OUT/shiftaddllm_eval_${name}.log"
  if [ -s "$target" ]; then
    say "SKIP  gpu$gpu evaluate $name (already present)"
    return 0
  fi
  say "START gpu$gpu evaluate $name"
  [ "$DRY" = 1 ] && return 0
  if CUDA_VISIBLE_DEVICES="$gpu" python experiments/pao_vs_bpla_model.py \
        --models gpt2 --backends exact shiftaddllm --scopes multiplication \
        --gpt2-sequence-length 256 --gpt2-target-tokens 300000 \
        --stream-metrics --device cuda --seed 0 \
        --shiftaddllm-weights "$WEIGHTS/$name.pt" \
        --output "$target.partial" >"$log" 2>&1; then
    mv "$target.partial" "$target"
    say "DONE  gpu$gpu evaluate $name ($(grep "perplexity=" "$log" | tail -1 | sed 's/^ *//'))"
  else
    say "FAIL  gpu$gpu evaluate $name -- tail of $log:"
    tail -6 "$log" | sed 's/^/    /'
    return 1
  fi
}

# A job is "opt:<config>:<seed>" (validation only) or "gpt2:<config>:<seed>".
run_job() {
  local gpu="$1" job="$2"
  local kind config seed
  IFS=':' read -r kind config seed <<<"$job"
  if [ "$kind" = opt ]; then
    quantize "$gpu" facebook/opt-125m opt125m "$config" "$seed"
  else
    quantize "$gpu" gpt2 gpt2 "$config" "$seed" && evaluate "$gpu" "gpt2_${config}_s${seed}"
  fi
}

# Longest first: every Acc. job (a refinement per column, several times the
# work of Lat., which refines once per eight columns) before every Lat. job.
# Dealt round-robin over N workers, the first N jobs are the long ones, so no
# worker ends up with two of them while another idles.
JOBS=()
for mode in acc lat; do
  JOBS+=("opt:${mode}3:0")
  for seed in "${SEEDS[@]}"; do
    for config in "${CONFIGS[@]}"; do
      if [ "${config%?}" = "$mode" ]; then JOBS+=("gpt2:${config}:${seed}"); fi
    done
  done
done

say "=== ShiftAddLLM run: ${#JOBS[@]} jobs on GPUs ${GPUS[*]} ==="
if [ "$DRY" != 1 ]; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee -a "$LOG"

  # Dependencies the quantizer imports that the image may lack. primefac is
  # used by the Acc. mode's random butterfly rotations.
  python -c "import primefac, scipy" 2>/dev/null || pip install -q primefac scipy \
    || { say "could not install primefac/scipy"; exit 1; }
  python -c "import transformers, datasets" 2>/dev/null || pip install -q -r requirements.txt \
    || { say "could not install requirements.txt"; exit 1; }

  ./fetch_shiftaddllm.sh | tee -a "$LOG" || { say "could not fetch ShiftAddLLM"; exit 1; }

  # Fetch checkpoints and data once; parallel first downloads into one HF
  # cache can corrupt it.
  python - <<'PY' || { say "checkpoint / dataset pre-fetch failed"; exit 1; }
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
for split in ("train", "test"):
    print(split, len(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)))
for model in ("gpt2", "facebook/opt-125m"):
    AutoTokenizer.from_pretrained(model)
    AutoModelForCausalLM.from_pretrained(model)
    print("ready:", model)
PY
fi

# Deal the jobs onto the GPUs round-robin; each GPU runs its share in order.
declare -a pids=()
for ((g = 0; g < ${#GPUS[@]}; g++)); do
  gpu="${GPUS[$g]}"
  mine=()
  for ((j = g; j < ${#JOBS[@]}; j += ${#GPUS[@]})); do
    mine+=("${JOBS[$j]}")
  done
  [ ${#mine[@]} -eq 0 ] && continue
  say "gpu$gpu <- ${mine[*]}"
  (
    rc=0
    for job in "${mine[@]}"; do run_job "$gpu" "$job" || rc=1; done
    exit $rc
  ) &
  pids+=($!)
done

failures=0
for p in "${pids[@]}"; do
  wait "$p" || failures=$(( failures + 1 ))
done

say "=== finished (${failures} workers reported a failure) ==="
say "summary:  python report_shiftaddllm.py"
exit $(( failures > 0 ))
