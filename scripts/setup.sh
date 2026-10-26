#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
if [ ! -d .venv ]; then
  "${PYTHON_BIN:-python3}" -m venv .venv
fi
exec .venv/bin/python -m pip install -r requirements.txt
