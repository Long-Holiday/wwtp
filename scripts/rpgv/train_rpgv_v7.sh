#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
config="${RPGV_V7_CONFIG:-configs/rpgv/v7/rpgv_v7.py}"
variant="$(basename "$config" .py)"
work_root="${RPGV_V7_WORK_ROOT:-work_dirs/rpgv_v7_${variant}_1m}"
if [[ "$variant" == "rpgv_v7" ]] && [[ -z "${RPGV_V7_WORK_ROOT:-}" ]]; then
  work_root="work_dirs/rpgv_v7_structure_1m"
fi
resume=false
for arg in "$@"; do
  if [[ "$arg" == "--resume" || "$arg" == --resume=* ]]; then
    resume=true
  fi
done
if [[ -e "$work_root" ]] && [[ "$resume" == false ]]; then
  echo "Refusing to overwrite $work_root. Use a new RPGV_V7_WORK_ROOT or --resume." >&2
  exit 1
fi
export WWTP_DATA_ROOT="${WWTP_DATA_ROOT:-${WWTP_V5_DATA_ROOT:-wwtp_semantic_dataset_1m}}"
export WWTP_PSEUDO_ROOT="${WWTP_PSEUDO_ROOT:-$WWTP_DATA_ROOT/pseudo_geometry}"
# Read the effective pipeline, including caller overrides, through the common preparer.
python - "$config" "$@" <<'PY'
from pathlib import Path
import sys
from scripts.run_experiments import prepare_geometry
arguments = sys.argv[2:]
overrides = []
if '--cfg-options' in arguments:
    for value in arguments[arguments.index('--cfg-options') + 1:]:
        if value.startswith('--'):
            break
        overrides.append(value)
prepare_geometry([Path(sys.argv[1])], 'wwtp', ['train', 'val'], overrides, False)
PY
python tools/train.py "$config" --work-dir "$work_root" --disable-early-stopping "$@"
