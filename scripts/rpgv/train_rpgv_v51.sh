#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
config="${RPGV_V51_CONFIG:-configs/rpgv/v51/rpgv_v51.py}"
work_root="${RPGV_V51_WORK_ROOT:-work_dirs/rpgv_v51_1m}"
python_bin="${PYTHON_BIN:-python}"
for arg in "$@"; do
  if [[ "$arg" == --resume* ]]; then
    exec "$python_bin" tools/train.py "$config" --work-dir "$work_root" "$@"
  fi
done
if [[ -e "$work_root" ]]; then
  echo "Work directory exists: $work_root. Use --resume or a fresh RPGV_V51_WORK_ROOT." >&2
  exit 1
fi
if [[ "$config" == *from_scratch.py ]]; then
  exec "$python_bin" tools/train.py "$config" --work-dir "$work_root" "$@"
fi
source_checkpoint="${RPGV_V51_SOURCE_CHECKPOINT:-}"
if [[ -z "$source_checkpoint" ]]; then
  shopt -s nullglob
  candidates=(work_dirs/rpgv_v5_1m/best_binary_Foreground_IoU_iter_*.pth)
  if [[ "${#candidates[@]}" -ne 1 ]]; then
    echo "Set RPGV_V51_SOURCE_CHECKPOINT to the validation-selected v5 checkpoint." >&2
    exit 1
  fi
  source_checkpoint="${candidates[0]}"
fi
"$python_bin" tools/rpgv/initialize_rpgv_v51.py "$source_checkpoint" "$work_root/init.pth" --config "$config"
RPGV_V51_INIT_CHECKPOINT="$work_root/init.pth" \
  "$python_bin" tools/train.py "$config" --work-dir "$work_root" --disable-early-stopping "$@"
