#!/usr/bin/env bash
set -euo pipefail
RPGV_V3_ROOT="${RPGV_V3_WORK_ROOT:-work_dirs/rpgv_v3_adapter}"
RPGV_V3_SOURCE="${RPGV_V3_RGB_CHECKPOINT:-work_dirs/rpgv_v2_staged/stage1_rgb/best_binary_Foreground_IoU_iter_36000.pth}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -e "${RPGV_V3_ROOT}" ]]; then
  echo "Directory exists: ${RPGV_V3_ROOT}. Resume explicitly or choose a new RPGV_V3_WORK_ROOT." >&2
  exit 1
fi
if [[ ! -f "${RPGV_V3_SOURCE}" ]]; then
  echo "Set RPGV_V3_RGB_CHECKPOINT to the validation-selected v2 RGB checkpoint." >&2
  exit 1
fi
"${PYTHON_BIN}" tools/initialize_rpgv_v3.py "${RPGV_V3_SOURCE}" "${RPGV_V3_ROOT}/init.pth"
RPGV_V3_INIT_CHECKPOINT="${RPGV_V3_ROOT}/init.pth" \
  "${PYTHON_BIN}" tools/train.py configs/v3/rpgv_v3_adapter.py --work-dir "${RPGV_V3_ROOT}"
