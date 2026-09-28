#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
config="${RPGV_V5_CONFIG:-configs/v5/rpgv_v5.py}"
work_root="${RPGV_V5_WORK_ROOT:-work_dirs/rpgv_v5_1m}"
if [[ -e "$work_root" ]] && [[ ! "$*" =~ "--resume" ]]; then
  echo "Refusing to overwrite $work_root. Use a new RPGV_V5_WORK_ROOT or tools/train.py --resume." >&2
  exit 1
fi
python tools/train.py "$config" --work-dir "$work_root" --disable-early-stopping "$@"
