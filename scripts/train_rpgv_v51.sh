#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

config="${RPGV_V51_CONFIG:-configs/v51/rpgv_v51.py}"
source_checkpoint="${RPGV_V51_SOURCE_CHECKPOINT:-work_dirs/rpgv_v5_1m/best_binary_Foreground_IoU_iter_12000.pth}"
work_root="${RPGV_V51_WORK_ROOT:-work_dirs/rpgv_v51_1m}"
python_bin="${PYTHON_BIN:-python}"

if [[ -e "$work_root" ]] && [[ ! "$*" =~ "--resume" ]]; then
  echo "Refusing to overwrite existing work root: $work_root" >&2
  exit 1
fi
if [[ ! -f "$config" ]]; then
  echo "Missing v5.1 config: $config" >&2
  exit 1
fi
if [[ ! -f "$source_checkpoint" ]]; then
  echo "Missing v5 source checkpoint: $source_checkpoint" >&2
  exit 1
fi
if [[ "$config" == *from_scratch.py ]]; then
  echo "from_scratch.py has no v5 migration; train it directly with tools/train.py." >&2
  exit 1
fi

"$python_bin" tools/initialize_rpgv_v51.py \
  "$source_checkpoint" "$work_root/init.pth" --config "$config"

RPGV_V51_INIT_CHECKPOINT="$work_root/init.pth" \
  "$python_bin" tools/train.py "$config" \
    --work-dir "$work_root" --disable-early-stopping "$@"
