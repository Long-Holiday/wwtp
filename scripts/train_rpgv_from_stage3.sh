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

stage2_dir="${WORK_ROOT}/stage2_geometry"
stage3_dir="${WORK_ROOT}/stage3_joint"

if [[ -z "${RPGV_STAGE2_CHECKPOINT:-}" ]]; then
  stage2_checkpoint="$(latest_checkpoint "${stage2_dir}")"
else
  stage2_checkpoint="${RPGV_STAGE2_CHECKPOINT}"
fi
printf 'Using Stage 2 Checkpoint: %s\n' "${stage2_checkpoint}"

# Run Stage 3 (Joint Fine-tuning, 40k iters)
printf '=== Starting Stage 3 Joint Training (40k iters) ===\n'
RPGV_STAGE2_CHECKPOINT="${stage2_checkpoint}" \
  "${PYTHON_BIN}" tools/train.py \
  configs/experiments/rpgv_stage3_joint.py \
  --work-dir "${stage3_dir}" "$@"
final_checkpoint="$(latest_checkpoint "${stage3_dir}")"
printf 'Stage 3 complete. Best checkpoint: %s\n' "${final_checkpoint}"

# Run Test Set Evaluation on final checkpoint
printf '=== Starting Test Set Evaluation ===\n'
"${PYTHON_BIN}" tools/test.py \
  configs/experiments/rpgv_stage3_joint.py \
  "${final_checkpoint}" \
  --work-dir "${stage3_dir}/test_eval"

printf 'RPGV-Net full staged workflow complete: %s\n' "${final_checkpoint}"
