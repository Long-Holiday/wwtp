#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
config="${SEGNEXT_CONFIG:-configs/experiments/segnext.py}"
work_root="${SEGNEXT_WORK_ROOT:-work_dirs/segnext_1m}"
if [[ -e "$work_root" ]] && [[ ! "$*" =~ "--resume" ]]; then
  echo "Refusing to overwrite $work_root. Use a new SEGNEXT_WORK_ROOT or tools/train.py --resume." >&2
  exit 1
fi
python tools/train.py "$config" --work-dir "$work_root" --disable-early-stopping "$@"
