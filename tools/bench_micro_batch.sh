#!/usr/bin/env bash
# Measure how large a micro batch each model can actually train at, and how fast,
# using the production training entry (tools/train.py) instead of a synthetic
# harness: the numbers include the real data pipeline, optimizer step and cuDNN
# autotuning. One row per attempt goes to $OUT/summary.tsv.
#
#   tools/bench_micro_batch.sh hrnet:configs/wwtp/hrnet.py segformer:configs/wwtp/segformer.py
#   WWTP_BENCH_DTYPE=float16 tools/bench_micro_batch.sh mask2former:configs/wwtp/mask2former.py
#
# Environment:
#   WWTP_BENCH_DTYPE     autocast dtype to probe (bfloat16 default, float16 for models
#                        whose CUDA kernels implement only fp16)
#   WWTP_BENCH_BATCHES   micro batches to try in order (default "8 4 2")
#   WWTP_BENCH_OUT       output directory (default /tmp/wwtp_micro_batch_probe)
#   WWTP_BENCH_TIMEOUT   seconds per attempt, default 100
#
# Each attempt is cut off by the timeout on purpose: mmengine always validates on the
# last iteration, so a run that finishes by itself would pay for a full validation
# pass. The train lines this needs are written long before that. Requires the WWTP
# dataset and the conda env on the training host.
set -u
cd "${WWTP_REPO_ROOT:-/root/wwtp}"
if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${WWTP_BENCH_ENV:-wwtp-v6}"
export WWTP_DATA_ROOT="${WWTP_DATA_ROOT:-/root/autodl-fs/wwtp_semantic_dataset_1024}"
export WWTP_PSEUDO_ROOT="${WWTP_PSEUDO_ROOT:-$WWTP_DATA_ROOT/pseudo_geometry}"
export WWTP_NUM_WORKERS="${WWTP_NUM_WORKERS:-8}"
export WWTP_VAL_NUM_WORKERS="${WWTP_VAL_NUM_WORKERS:-4}"
export PYTHONUNBUFFERED=1

dtype="${WWTP_BENCH_DTYPE:-bfloat16}"
batches="${WWTP_BENCH_BATCHES:-8 4 2}"
OUT="${WWTP_BENCH_OUT:-/tmp/wwtp_micro_batch_probe}"
timeout_seconds="${WWTP_BENCH_TIMEOUT:-100}"
loss_scale=1.0
[ "$dtype" = float16 ] && loss_scale=dynamic

mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
printf 'model\tbatch\tstatus\tseconds_per_iter\tsteady_mib\tpeak_mib\n' > "$SUMMARY"

for entry in "$@"; do
  model="${entry%%:*}"
  config="${entry#*:}"
  for batch in $batches; do
    log="$OUT/${model}_${dtype}_bs${batch}.log"
    echo "[$(date '+%H:%M:%S')] $model batch $batch $dtype"
    timeout "$timeout_seconds" python tools/train.py "$config" \
      --work-dir "$OUT/run_${model}_bs${batch}" \
      --no-tensorboard --no-pretrained \
      --cfg-options \
        train_dataloader.batch_size="$batch" \
        optim_wrapper.type=AmpOptimWrapper \
        optim_wrapper.dtype="$dtype" \
        optim_wrapper.loss_scale="$loss_scale" \
        optim_wrapper.accumulative_counts=1 \
        train_cfg.max_iters=1000000 \
        train_cfg.val_interval=1000000 \
        default_hooks.logger.interval=5 \
        default_hooks.checkpoint.interval=1000000 \
      > "$log" 2>&1
    if grep -q "out of memory\|OutOfMemoryError" "$log"; then
      printf '%s\t%s\tOOM\t\t\t\n' "$model" "$batch" >> "$SUMMARY"
      rm -rf "$OUT/run_${model}_bs${batch}"
      continue
    fi
    if ! grep -q "Iter(train)" "$log"; then
      printf '%s\t%s\tno-train-line\t\t\t\n' "$model" "$batch" >> "$SUMMARY"
      tail -3 "$log" | sed 's/^/    /'
      continue
    fi
    memory=$(grep -o "memory: [0-9]*" "$log" | awk '{print $2}')
    steady=$(echo "$memory" | tail -1)
    peak=$(echo "$memory" | sort -n | tail -1)
    seconds=$(grep "Iter(train)" "$log" | grep -oE "[^_]time: [0-9.]+" | tail -1 | awk '{print $2}')
    printf '%s\t%s\tok\t%s\t%s\t%s\n' "$model" "$batch" "$seconds" "$steady" "$peak" >> "$SUMMARY"
    rm -rf "$OUT/run_${model}_bs${batch}"
    break
  done
done
cat "$SUMMARY"
