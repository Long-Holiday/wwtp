#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
work_root="${RPGV_V3_WORK_ROOT:-work_dirs/rpgv_v3_adapter}"
python_bin="${PYTHON_BIN:-python}"
for arg in "$@"; do
  if [[ "$arg" == --resume* ]]; then
    exec "$python_bin" tools/train.py configs/rpgv/v3/rpgv_v3_adapter.py --work-dir "$work_root" "$@"
  fi
done
if [[ -e "$work_root" ]]; then
  echo "Work directory exists: $work_root. Use --resume or a fresh RPGV_V3_WORK_ROOT." >&2
  exit 1
fi
source_checkpoint="${RPGV_V3_RGB_CHECKPOINT:-}"
if [[ -z "$source_checkpoint" || ! -f "$source_checkpoint" ]]; then
  echo "Set RPGV_V3_RGB_CHECKPOINT to a validation-selected v2 RGB checkpoint trained on the same 1024 data." >&2
  exit 1
fi
"$python_bin" tools/rpgv/initialize_rpgv_v3.py "$source_checkpoint" "$work_root/init.pth"
RPGV_V3_INIT_CHECKPOINT="$work_root/init.pth" \
  "$python_bin" tools/train.py configs/rpgv/v3/rpgv_v3_adapter.py --work-dir "$work_root" "$@"
