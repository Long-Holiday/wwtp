#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
config="${RPGV_V6_CONFIG:-configs/rpgv/v6/rpgv_v6.py}"
variant="$(basename "$config" .py)"
work_root="${RPGV_V6_WORK_ROOT:-work_dirs/rpgv_v6_siglip_${variant}_1m}"
if [[ "$variant" == "rpgv_v6" ]] && [[ -z "${RPGV_V6_WORK_ROOT:-}" ]]; then
  work_root="work_dirs/rpgv_v6_siglip_1m"
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
data_root="${WWTP_DATA_ROOT:-${WWTP_V5_DATA_ROOT:-wwtp_semantic_dataset_1m}}"
export WWTP_DATA_ROOT="$data_root"
export WWTP_V5_DATA_ROOT="$data_root"
export WWTP_PSEUDO_ROOT="$data_root/pseudo_geometry"

shopt -s nullglob
missing_geometry=false
for split in train val test; do
  images=("$data_root/images/$split/"*.png)
  if [[ ${#images[@]} -eq 0 ]]; then
    echo "No PNG images found in $data_root/images/$split" >&2
    exit 1
  fi
  for image in "${images[@]}"; do
    image_name="${image##*/}"
    if [[ ! -f "$WWTP_PSEUDO_ROOT/$split/${image_name%.png}.npz" ]]; then
      missing_geometry=true
      break
    fi
  done
done
shopt -u nullglob

if [[ "$missing_geometry" == true ]]; then
  echo "Generating missing pseudo geometry in $WWTP_PSEUDO_ROOT"
  python tools/data/generate_pseudo_geometry.py "$data_root" \
    --output-root "$WWTP_PSEUDO_ROOT" --splits train val test
else
  echo "Pseudo geometry is complete in $WWTP_PSEUDO_ROOT; skipping generation."
fi

python tools/train.py "$config" --work-dir "$work_root" --disable-early-stopping "$@"
