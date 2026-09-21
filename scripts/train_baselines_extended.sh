#!/usr/bin/env bash
# ==============================================================================
# Script to train 8 mainstream baseline models with extended training iterations
# resuming from their original iter_3000.pth checkpoints.
# ==============================================================================
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
MODELS=(
  "deeplabv3plus"
  "hrnet"
  "mask2former"
  "rs_mamba"
  "segformer"
  "segnext"
  "unet"
  "unetformer"
)

latest_or_best_checkpoint() {
  local dir="$1"
  local ckpt
  # 1. Look for explicit best checkpoint saved by CheckpointHook
  ckpt="$(find "${dir}" -maxdepth 1 -type f \
    -name 'best_binary_Foreground_IoU_iter_*.pth' -printf '%T@ %p\n' 2>/dev/null \
    | sort -n | tail -n 1 | cut -d' ' -f2-)"
  
  # 2. If not found, parse scalars.json to find the iteration with peak Foreground_IoU
  if [[ -z "${ckpt}" ]]; then
    ckpt="$("${PYTHON_BIN}" -c "
import glob, json, os, sys
dir_path = sys.argv[1]
scalars = glob.glob(os.path.join(dir_path, '*', 'vis_data', 'scalars.json'))
best_iter, best_val = None, -1.0
for sc in scalars:
    with open(sc) as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'binary/Foreground_IoU' in data and 'step' in data:
                    v = float(data['binary/Foreground_IoU'])
                    s = int(data['step'])
                    if v > best_val:
                        best_val = v
                        best_iter = s
            except Exception:
                pass
if best_iter:
    target = os.path.join(dir_path, f'iter_{best_iter}.pth')
    if os.path.exists(target):
        print(target)
" "${dir}" 2>/dev/null || true)"
  fi

  # 3. Fallback to latest iter_*.pth
  if [[ -z "${ckpt}" ]]; then
    ckpt="$(find "${dir}" -maxdepth 1 -type f \
      -name 'iter_*.pth' -printf '%T@ %p\n' 2>/dev/null \
      | sort -n | tail -n 1 | cut -d' ' -f2-)"
  fi
  printf '%s\n' "${ckpt}"
}

printf '====================================================================\n'
printf ' Starting Extended Training for 8 Mainstream Models (Resuming from 3k)\n'
printf '====================================================================\n'

for model in "${MODELS[@]}"; do
  orig_dir="work_dirs/${model}"
  extend_dir="work_dirs/${model}_extended"
  orig_ckpt="${orig_dir}/iter_3000.pth"
  cfg_path="configs/experiments_extended/${model}.py"

  printf '\n>>> [%s] Checking prerequisite checkpoint...\n' "${model}"
  if [[ ! -f "${orig_ckpt}" ]]; then
    printf 'Error: Baseline checkpoint %s not found! Skipping %s.\n' "${orig_ckpt}" "${model}" >&2
    continue
  fi

  printf '>>> [%s] Checkpoint found: %s\n' "${model}" "${orig_ckpt}"
  printf '>>> [%s] Extended config: %s\n' "${model}" "${cfg_path}"
  printf '>>> [%s] Target work-dir: %s\n' "${model}" "${extend_dir}"

  if [[ -d "${extend_dir}/test_results" ]] && find "${extend_dir}/test_results" -name "*.json" | grep -q json; then
    printf '>>> [%s] Already evaluated in %s. Skipping.\n' "${model}" "${extend_dir}/test_results"
    continue
  fi

  mkdir -p "${extend_dir}"

  # Step 1: Train from iter 3000 to extended iters (20k)
  printf '>>> [%s] Launching extended training...\n' "${model}"
  "${PYTHON_BIN}" tools/train.py \
    "${cfg_path}" \
    --work-dir "${extend_dir}" \
    --resume "${orig_ckpt}" \
    --no-resume-scheduler

  # Step 2: Identify best checkpoint for evaluation
  best_ckpt="$(latest_or_best_checkpoint "${extend_dir}")"
  if [[ -z "${best_ckpt}" ]]; then
    printf 'Error: No checkpoint found in %s after training!\n' "${extend_dir}" >&2
    continue
  fi
  printf '>>> [%s] Training completed. Best checkpoint: %s\n' "${model}" "${best_ckpt}"

  # Step 3: Run test set evaluation
  printf '>>> [%s] Evaluating best checkpoint on test set...\n' "${model}"
  "${PYTHON_BIN}" tools/test.py \
    "${cfg_path}" \
    "${best_ckpt}" \
    --work-dir "${extend_dir}/test_results"

  printf '>>> [%s] Evaluation finished successfully!\n' "${model}"
done

printf '\n====================================================================\n'
printf ' All 8 Mainstream Models Extended Training & Evaluation Complete!\n'
printf '====================================================================\n'
