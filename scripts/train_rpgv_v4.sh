#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
RPGV_V4_ROOT="${RPGV_V4_WORK_ROOT:-work_dirs/rpgv_v4}"
if [[ -e "${RPGV_V4_ROOT}" ]]; then
  echo "Directory exists: ${RPGV_V4_ROOT}. Resume with tools/train.py --resume or set a new RPGV_V4_WORK_ROOT." >&2
  exit 1
fi
exec "${PYTHON_BIN}" tools/train.py configs/v4/rpgv_v4.py --work-dir "${RPGV_V4_ROOT}" "$@"
