#!/bin/sh
set -eu
: "${ADACGR_U_REFERENCE:?Set ADACGR_U_REFERENCE to the private reference package}"
: "${REMOTE_OUTPUT_ROOT:?Set REMOTE_OUTPUT_ROOT to an absolute output path}"

PHASE=${1:-smoke}
DATASET=${2:-Beauty}
SEED=${3:-2025}
FOLD=${4:-early}
PROFILE=${5:-baseline}
SELECTION=${6:-}

ARGS="--dataset $DATASET --seed $SEED --phase $PHASE --profile $PROFILE --output-root $REMOTE_OUTPUT_ROOT"
if [ "$PHASE" != full ]; then ARGS="$ARGS --fold $FOLD"; fi
if [ -n "$SELECTION" ]; then ARGS="$ARGS --selection $SELECTION"; fi
# shellcheck disable=SC2086
exec "${PYTHON_BIN:-python3}" "$ADACGR_U_REFERENCE/run_u.py" $ARGS
