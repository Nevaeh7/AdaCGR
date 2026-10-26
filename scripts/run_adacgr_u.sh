#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ "$#" -gt 0 ]; then INPUT=$1; shift; else INPUT="$ROOT/scripts/fixtures/adacgr_u_toy.json"; fi
if [ "$#" -gt 0 ]; then OUTPUT=$1; shift; else OUTPUT="$ROOT/outputs/adacgr_u_toy.json"; fi
cd "$ROOT"
exec "${PYTHON_BIN:-python3}" -m scripts.run_adacgr_u \
  --input "$INPUT" \
  --output "$OUTPUT" \
  "$@"
