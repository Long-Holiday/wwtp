#!/usr/bin/env bash
# ==============================================================================
# Script to evaluate the best checkpoint from RPGV Stage 3 continuation training
# on the official test set and save results in work_dirs/rpgv_stage3_restart_30k/test_eval.
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

WORK_DIR="work_dirs/rpgv_stage3_restart_30k"
CONFIG="configs/experiments/rpgv_stage3_joint.py"
TEST_EVAL_DIR="${WORK_DIR}/test_eval"

echo "===================================================================="
echo " Evaluating RPGV Stage 3 Restart Checkpoint on Test Set"
echo "===================================================================="

# 1. Find the best checkpoint saved by CheckpointHook
BEST_CKPT="$(find "${WORK_DIR}" -maxdepth 1 -type f -name 'best_binary_Foreground_IoU_iter_*.pth' 2>/dev/null | sort -V | tail -n 1 || true)"

# 2. Fallback to latest iter_*.pth if no best checkpoint is found
if [[ -z "${BEST_CKPT}" ]]; then
  BEST_CKPT="$(find "${WORK_DIR}" -maxdepth 1 -type f -name 'iter_*.pth' 2>/dev/null | sort -V | tail -n 1 || true)"
fi

if [[ -z "${BEST_CKPT}" ]]; then
  echo "Error: No checkpoint found in ${WORK_DIR}!" >&2
  exit 1
fi

echo "Target WorkDir:    ${WORK_DIR}"
echo "Config File:       ${CONFIG}"
echo "Best Checkpoint:   ${BEST_CKPT}"
echo "Test Output Dir:   ${TEST_EVAL_DIR}"
echo "===================================================================="

mkdir -p "${TEST_EVAL_DIR}"

docker compose run --rm -e PYTHONUNBUFFERED=1 wwtp \
  python tools/test.py "${CONFIG}" "${BEST_CKPT}" \
  --work-dir "${TEST_EVAL_DIR}"

echo "===================================================================="
echo " Evaluation Completed Successfully!"
echo " Results stored in ${TEST_EVAL_DIR}"
echo "===================================================================="
