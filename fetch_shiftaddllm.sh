#!/usr/bin/env bash
# Clone the ShiftAddLLM authors' code (Apache-2.0) at the commit every reported
# number was produced with. The quantizer is imported from this clone
# unchanged; see modules/torch_shiftaddllm.py. The clone stays out of git.
#
#   ./fetch_shiftaddllm.sh                    # into third_party/ShiftAddLLM
#   SHIFTADDLLM_ROOT=/elsewhere ./fetch_shiftaddllm.sh

set -eu

cd "$(dirname "$0")"

URL=https://github.com/GATECH-EIC/ShiftAddLLM.git
COMMIT=1053837ae320392e6cf5611ab3378550a1f6ba11
DEST="${SHIFTADDLLM_ROOT:-third_party/ShiftAddLLM}"

if [ -d "$DEST/.git" ]; then
  current=$(git -C "$DEST" rev-parse HEAD)
  if [ "$current" = "$COMMIT" ]; then
    echo "ShiftAddLLM already at ${COMMIT:0:12} in $DEST"
    exit 0
  fi
  echo "ShiftAddLLM in $DEST is at ${current:0:12}; checking out ${COMMIT:0:12}"
  git -C "$DEST" fetch --quiet origin
else
  mkdir -p "$(dirname "$DEST")"
  git clone --quiet "$URL" "$DEST"
fi
git -C "$DEST" checkout --quiet "$COMMIT"
echo "ShiftAddLLM at $(git -C "$DEST" rev-parse --short=12 HEAD) in $DEST"
