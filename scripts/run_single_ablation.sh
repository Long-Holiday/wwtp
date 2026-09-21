#!/usr/bin/env bash
set -euo pipefail

ablation="${1:?Usage: $0 <ablation_name> [stage2_checkpoint]}"
stage2_ckpt="${2:-work_dirs/rpgv_staged/stage2_geometry/best_binary_Foreground_IoU_iter_10000.pth}"
work_root="${RPGV_ABLATION_WORK_ROOT:-work_dirs/rpgv_ablations}"
ablation_dir="${work_root}/${ablation}/stage3_joint"
config="configs/ablations/rpgv_${ablation}.py"

if [[ ! -f "${config}" ]]; then
  echo "Error: config ${config} not found" >&2
  exit 1
fi

mkdir -p "${ablation_dir}"

echo "=================================================="
echo " Running Ablation: ${ablation}"
echo " Config: ${config}"
echo " Work dir: ${ablation_dir}"
echo " Stage 2 Checkpoint: ${stage2_ckpt}"
echo "=================================================="

# Check if already trained and evaluated
if ls "${ablation_dir}/best_binary_Foreground_IoU_iter_"*.pth 1>/dev/null 2>&1 && \
   ls "${ablation_dir}/test_eval/"*/*.json 1>/dev/null 2>&1; then
  echo "Ablation ${ablation} already trained and evaluated. Skipping training."
else
  docker compose run --rm \
    -e RPGV_STAGE2_CHECKPOINT="${stage2_ckpt}" \
    wwtp python tools/train.py "${config}" \
    --work-dir "${ablation_dir}" \
    --cfg-options \
      train_dataloader.batch_size=1 \
      train_dataloader.num_workers=4 \
      val_dataloader.num_workers=4 \
      optim_wrapper.accumulative_counts=8 \
      train_cfg.max_iters=16000 \
      train_cfg.val_interval=1000 \
      param_scheduler.0.end=750 \
      param_scheduler.1.begin=750 \
      param_scheduler.1.end=16000
fi

# Find best checkpoint
best_ckpt="$(find "${ablation_dir}" -maxdepth 1 -name "best_binary_Foreground_IoU_iter_*.pth" | sort -V | tail -n 1)"
if [[ -z "${best_ckpt}" ]]; then
  best_ckpt="$(find "${ablation_dir}" -maxdepth 1 -name "iter_*.pth" | sort -V | tail -n 1)"
fi

echo "Best checkpoint: ${best_ckpt}"

# Test set evaluation
test_eval_dir="${ablation_dir}/test_eval"
if ls "${test_eval_dir}/"*/*.json 1>/dev/null 2>&1; then
  echo "Test evaluation already exists in ${test_eval_dir}"
else
  echo "Evaluating best checkpoint on test set..."
  docker compose run --rm wwtp python tools/test.py \
    "${config}" \
    "${best_ckpt}" \
    --work-dir "${test_eval_dir}"
fi

echo "Ablation ${ablation} completed successfully!"
