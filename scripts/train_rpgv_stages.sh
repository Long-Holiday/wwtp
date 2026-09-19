#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${RPGV_WORK_ROOT:-work_dirs/rpgv_staged}"
PYTHON_BIN="${PYTHON_BIN:-python}"

latest_checkpoint() {
  local work_dir="$1"
  local checkpoint
  checkpoint="$(find "${work_dir}" -maxdepth 1 -type f \
    -name 'best_binary_Foreground_IoU_iter_*.pth' -printf '%T@ %p\n' \
    | sort -n | tail -n 1 | cut -d' ' -f2-)"
  if [[ -z "${checkpoint}" ]]; then
    checkpoint="$(find "${work_dir}" -maxdepth 1 -type f \
      -name 'iter_*.pth' -printf '%T@ %p\n' \
      | sort -n | tail -n 1 | cut -d' ' -f2-)"
  fi
  if [[ -z "${checkpoint}" ]]; then
    echo "No checkpoint found in ${work_dir}" >&2
    return 1
  fi
  printf '%s\n' "${checkpoint}"
}

stage1_dir="${WORK_ROOT}/stage1_rgb"
stage2_dir="${WORK_ROOT}/stage2_geometry"
stage3_dir="${WORK_ROOT}/stage3_joint"

"${PYTHON_BIN}" tools/train.py \
  configs/experiments/rpgv_stage1_rgb.py \
  --work-dir "${stage1_dir}" "$@"
stage1_checkpoint="$(latest_checkpoint "${stage1_dir}")"

RPGV_STAGE1_CHECKPOINT="${stage1_checkpoint}" \
  "${PYTHON_BIN}" tools/train.py \
  configs/experiments/rpgv_stage2_geometry.py \
  --work-dir "${stage2_dir}" "$@"
stage2_checkpoint="$(latest_checkpoint "${stage2_dir}")"

RPGV_STAGE2_CHECKPOINT="${stage2_checkpoint}" \
  "${PYTHON_BIN}" tools/train.py \
  configs/experiments/rpgv_stage3_joint.py \
  --work-dir "${stage3_dir}" "$@"

final_checkpoint="$(latest_checkpoint "${stage3_dir}")"
printf 'RPGV-Net staged training complete: %s\n' "${final_checkpoint}"
