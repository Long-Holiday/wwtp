#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
config="${RPGV_V6_CONFIG:-configs/rpgv/v6/rpgv_v6.py}"
variant="$(basename "$config" .py)"
work_root="${RPGV_V6_WORK_ROOT:-work_dirs/rpgv_v6_${variant}_1m}"
if [[ "$variant" == "rpgv_v6" ]] && [[ -z "${RPGV_V6_WORK_ROOT:-}" ]]; then
  work_root="work_dirs/rpgv_v6_1m"
fi
resume=false
for arg in "$@"; do
  if [[ "$arg" == "--resume" || "$arg" == --resume=* ]]; then
    resume=true
  fi
done
if [[ -e "$work_root" ]] && [[ "$resume" == false ]]; then
  echo "Refusing to overwrite $work_root. Use a new RPGV_V6_WORK_ROOT or --resume." >&2
  exit 1
fi
python tools/train.py "$config" --work-dir "$work_root" --disable-early-stopping "$@"
