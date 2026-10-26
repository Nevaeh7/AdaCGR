#!/bin/sh
set -eu
: "${ADACGR_X_REFERENCE:?Set ADACGR_X_REFERENCE to the private reference package}"
: "${REMOTE_OUTPUT_ROOT:?Set REMOTE_OUTPUT_ROOT to an absolute output path}"

LEARNER=${1:?learner is required}
DATASET=${2:-Beauty}
FOLD=${3:-early}
BANK=${4:-b1}
SEED=${5:-2025}

case "$LEARNER" in
  ft|lsat|reformer|peso|dact) ;;
  *) echo "unsupported learner: $LEARNER" >&2; exit 2 ;;
esac

ARGS="--dataset $DATASET --learner $LEARNER --fold $FOLD --bank $BANK --seed $SEED --output-root $REMOTE_OUTPUT_ROOT"
if [ "${SMOKE:-1}" = 1 ]; then ARGS="$ARGS --smoke"; fi
# shellcheck disable=SC2086
exec "${PYTHON_BIN:-python3}" "$ADACGR_X_REFERENCE/run_screen.py" $ARGS
