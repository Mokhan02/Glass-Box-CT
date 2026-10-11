#!/usr/bin/env bash
# Extract activations for every fully fetched train scan, then wait for more, until TARGET are done.
#   PY=/path/to/venv/bin/python bash scripts/extract_loop.sh [TARGET=400] [OUT=cache/acts] [extra args, e.g. --random-init]
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TARGET="${1:-400}"
PY="${PY:?set PY to the venv python}"
OUT="${2:-cache/acts}"
if [ $# -ge 2 ]; then shift 2; else shift $#; fi
while true; do
  "$PY" scripts/extract_activations.py --root cache/ts --out "$OUT" --device cuda "$@" 2>&1 | grep --line-buffered -v -i warn
  n=$(ls "$OUT" 2>/dev/null | grep -c '\.npz$')
  echo "extracted $n/$TARGET"
  [ "$n" -ge "$TARGET" ] && break
  sleep 30
done
