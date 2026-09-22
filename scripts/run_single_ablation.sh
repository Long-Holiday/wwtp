#!/usr/bin/env bash
set -euo pipefail

ablation="${1:?Usage: $0 <ablation_name> [stage2_checkpoint]}"
stage2_ckpt="${2:-work_dirs/rpgv_staged/stage2_geometry/best_binary_Foreground_IoU_iter_10000.pth}"
work_root="${RPGV_ABLATION_WORK_ROOT:-work_dirs/rpgv_ablations}"
python_bin="${PYTHON_BIN:-python3}"

# Reuse the suite's 16k protocol and its incomplete-run guard.
exec "${python_bin}" scripts/run_ablation_suite.py \
  --ablation "${ablation}" \
  --stage2-ckpt "${stage2_ckpt}" \
  --work-root "${work_root}"
