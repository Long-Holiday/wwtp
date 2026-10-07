#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ablation="${1:?Usage: $0 <ablation_name> <stage2_checkpoint>}"
stage2_ckpt="${2:-${RPGV_STAGE2_CHECKPOINT:-}}"
if [[ -z "$stage2_ckpt" || ! -f "$stage2_ckpt" ]]; then
  echo "Supply a validation-selected Stage 2 checkpoint trained on the 1024 dataset." >&2
  exit 1
fi
work_root="${RPGV_ABLATION_WORK_ROOT:-work_dirs/rpgv_ablations}"
python_bin="${PYTHON_BIN:-python3}"

# Reuse the suite's 16k protocol and its incomplete-run guard.
exec "${python_bin}" scripts/run_ablation_suite.py \
  --ablation "${ablation}" \
  --stage2-ckpt "${stage2_ckpt}" \
  --work-root "${work_root}"
