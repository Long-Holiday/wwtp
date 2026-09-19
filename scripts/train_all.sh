#!/usr/bin/env bash
set -euo pipefail

MODELS=(unet deeplabv3plus hrnet segformer segnext mask2former unetformer rs_mamba)
GPU_COUNT="${GPU_COUNT:-1}"

for model in "${MODELS[@]}"; do
  config="configs/experiments/${model}.py"
  work_dir="work_dirs/${model}"
  if [[ "${GPU_COUNT}" -gt 1 ]]; then
    torchrun --nproc-per-node="${GPU_COUNT}" tools/train.py \
      "${config}" --launcher pytorch --work-dir "${work_dir}"
  else
    python tools/train.py "${config}" --work-dir "${work_dir}"
  fi
done
