#!/usr/bin/env bash
set -euo pipefail

# Fast protocol for an already trained RPGV model: initialize every ablation
# from the same stage-2 checkpoint and rerun only joint fine-tuning.
SOURCE_ROOT="${RPGV_SOURCE_WORK_ROOT:-work_dirs/rpgv_staged}"
WORK_ROOT="${RPGV_ABLATION_WORK_ROOT:-work_dirs/rpgv_ablations_stage3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_ABLATIONS=(
  no_global_context
  no_rgr
  no_depth_rectification
  offline_reliability_only
  no_frequency_validation
  no_boundary_fusion
  no_region_fusion
  unweighted_fusion
  no_boundary_refinement
  no_detail_refinement
  no_geometry_dropout
  no_shape_auxiliary
)

latest_checkpoint() {
  local work_dir="$1"
  local checkpoint
  checkpoint="$(find "${work_dir}" -maxdepth 1 -type f \
    -name 'best_binary_Foreground_IoU_iter_*.pth' -printf '%T@ %p\n' \
    | sort -n | tail -n 1 | cut -d' ' -f2-)"
  if [[ -z "${checkpoint}" ]]; then
    checkpoint="$(find "${work_dir}" -maxdepth 1 -type f \
      -name 'iter_*.pth' -printf '%T@ %p\n' \
      | sort -n | tail -n 1 | cut -d' ' -f2-)"
  fi
  if [[ -z "${checkpoint}" ]]; then
    printf 'No checkpoint found in %s\n' "${work_dir}" >&2
    return 1
  fi
  printf '%s\n' "${checkpoint}"
}

stage2_checkpoint="${RPGV_STAGE2_CHECKPOINT:-}"
if [[ -z "${stage2_checkpoint}" ]]; then
  stage2_checkpoint="$(latest_checkpoint "${SOURCE_ROOT}/stage2_geometry")"
fi
printf 'Using shared Stage 2 checkpoint: %s\n' "${stage2_checkpoint}"

if [[ -n "${RPGV_ABLATIONS:-}" ]]; then
  read -r -a selected_ablations <<< "${RPGV_ABLATIONS}"
else
  selected_ablations=("${DEFAULT_ABLATIONS[@]}")
fi

for ablation in "${selected_ablations[@]}"; do
  if [[ "${ablation}" == "full" ]]; then
    config="configs/experiments/rpgv_stage3_joint.py"
  else
    config="configs/ablations/rpgv_${ablation}.py"
  fi
  if [[ ! -f "${config}" ]]; then
    printf 'Unknown RPGV ablation or missing config: %s\n' "${ablation}" >&2
    exit 2
  fi

  printf 'Training Stage 3 ablation: %s\n' "${ablation}"
  RPGV_STAGE2_CHECKPOINT="${stage2_checkpoint}" \
    "${PYTHON_BIN}" tools/train.py "${config}" \
    --work-dir "${WORK_ROOT}/${ablation}/stage3_joint" "$@"
done
