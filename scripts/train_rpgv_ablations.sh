#!/usr/bin/env bash
set -euo pipefail

# Retrain every selected variant through the same three stages.  Keeping the
# stage schedule fixed is the recommended, fair comparison; the standalone
# configs/ablations files are intended for quicker stage-3 diagnostics.
WORK_ROOT="${RPGV_ABLATION_WORK_ROOT:-work_dirs/rpgv_ablations}"
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

if [[ -n "${RPGV_ABLATIONS:-}" ]]; then
  read -r -a selected_ablations <<< "${RPGV_ABLATIONS}"
else
  selected_ablations=("${DEFAULT_ABLATIONS[@]}")
fi

for ablation in "${selected_ablations[@]}"; do
  case "${ablation}" in
    full)
      overrides=()
      ;;
    no_global_context)
      overrides=(model.component_cfg.global_context=False)
      ;;
    no_rgr)
      overrides=(
        model.component_cfg.depth_rectification=False
        model.component_cfg.learned_reliability=False
      )
      ;;
    no_depth_rectification)
      overrides=(model.component_cfg.depth_rectification=False)
      ;;
    offline_reliability_only)
      overrides=(model.component_cfg.learned_reliability=False)
      ;;
    no_frequency_validation)
      overrides=(model.component_cfg.frequency_validation=False)
      ;;
    no_boundary_fusion)
      overrides=(model.component_cfg.boundary_fusion=False)
      ;;
    no_region_fusion)
      overrides=(model.component_cfg.region_fusion=False)
      ;;
    unweighted_fusion)
      overrides=(model.component_cfg.reliability_weighting=False)
      ;;
    no_boundary_refinement)
      overrides=(model.component_cfg.boundary_refinement=False)
      ;;
    no_detail_refinement)
      overrides=(model.component_cfg.detail_refinement=False)
      ;;
    no_geometry_dropout)
      overrides=(model.geometry_dropout_prob=0.0)
      ;;
    no_shape_auxiliary)
      overrides=(model.loss_weights.boundary=0.0 model.loss_weights.sdf=0.0)
      ;;
    *)
      printf 'Unknown RPGV ablation: %s\n' "${ablation}" >&2
      exit 2
      ;;
  esac

  printf 'Training RPGV ablation: %s\n' "${ablation}"
  if ((${#overrides[@]})); then
    RPGV_WORK_ROOT="${WORK_ROOT}/${ablation}" \
      bash scripts/train_rpgv_stages.sh "$@" \
      --cfg-options "${overrides[@]}"
  else
    RPGV_WORK_ROOT="${WORK_ROOT}/${ablation}" \
      bash scripts/train_rpgv_stages.sh "$@"
  fi
done
