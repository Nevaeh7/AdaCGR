#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
LEARNER=${1:?learner is required}
shift
case "$LEARNER" in
  ft|lsat|reformer|peso|dact) ;;
  *) echo "unsupported learner: $LEARNER" >&2; exit 2 ;;
esac

if [ "$#" -gt 0 ]; then INPUT=$1; shift; else INPUT="$ROOT/scripts/fixtures/adacgr_x_toy.json"; fi
if [ "$#" -gt 0 ]; then OUTPUT=$1; shift; else OUTPUT="$ROOT/outputs/adacgr_${LEARNER}_toy.json"; fi

cd "$ROOT"
exec "${PYTHON_BIN:-python3}" -m scripts.run_adacgr_learner \
  --learner "$LEARNER" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --min-feedback "${MIN_FEEDBACK:-2}" \
  --min-budget "${MIN_BUDGET:-2}" \
  "$@"
