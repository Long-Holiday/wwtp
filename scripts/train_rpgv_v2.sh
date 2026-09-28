#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RPGV_V2_ROOT="${RPGV_V2_WORK_ROOT:-work_dirs/rpgv_v2_joint_fixed}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -e "${RPGV_V2_ROOT}" ]]; then
  echo "Work directory already exists: ${RPGV_V2_ROOT}. Choose a fresh RPGV_V2_WORK_ROOT, or resume this run with tools/train.py configs/experiments/rpgv_v2_joint.py --work-dir ${RPGV_V2_ROOT} --resume." >&2
  exit 1
fi
exec "${PYTHON_BIN}" tools/train.py configs/experiments/rpgv_v2_joint.py --work-dir "${RPGV_V2_ROOT}"
