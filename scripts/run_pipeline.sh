#!/usr/bin/env bash
# Sequential pipeline for WWTP 1024 experiments:
# Phase 1: RPGV v7 train and test evaluation
# Phase 2: Baselines train and test evaluation, minus mask2former and hdnet
# Phase 2.1: Mask2Former
# Phase 2.2: HDNet
# Phase 2.4: RPGV v8 train and test evaluation
# Phase 2.5: RPGV v7 fresh 30k retraining, overwriting the previous result
# Phase 3: V7 ablations train and test evaluation
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -f /etc/network_turbo ]]; then
  source /etc/network_turbo
fi

conda_exe="${CONDA_EXE:-/root/miniconda3/bin/conda}"
if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate wwtp-v6
fi

export WWTP_DATA_ROOT="${WWTP_DATA_ROOT:-/root/autodl-fs/wwtp_semantic_dataset_1024}"
export WWTP_PSEUDO_ROOT="${WWTP_PSEUDO_ROOT:-$WWTP_DATA_ROOT/pseudo_geometry}"
export WWTP_NUM_WORKERS="${WWTP_NUM_WORKERS:-8}"
export WWTP_VAL_NUM_WORKERS="${WWTP_VAL_NUM_WORKERS:-4}"
export PYTHONUNBUFFERED=1

mkdir -p logs

# Phases 1 and 2 yield no new output once their models are trained, and --resume
# makes mmengine advance the dataloader over the whole remaining schedule (30000
# batches of decoded 1024px crops) before the loop exits, so restarting them
# costs a dataloader pass per model. Set WWTP_SKIP_COMPLETED_PHASES=1 when
# restarting a pipeline whose RPGV v7 and baseline outputs already exist.
skip_completed="${WWTP_SKIP_COMPLETED_PHASES:-0}"

if [[ "$skip_completed" == "1" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skipping Phase 1 (RPGV v7): already trained and evaluated." | tee -a logs/pipeline.log
else
  echo "==========================================================" | tee -a logs/pipeline.log
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Phase 1: RPGV v7" | tee -a logs/pipeline.log
  echo "==========================================================" | tee -a logs/pipeline.log
  python scripts/run_experiments.py train-test --dataset wwtp --suite v7 --resume 2>&1 | tee -a logs/pipeline_v7.log
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 1 Complete (RPGV v7)" | tee -a logs/pipeline.log
fi

if [[ "$skip_completed" == "1" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skipping Phase 2 (Baselines): already trained and evaluated." | tee -a logs/pipeline.log
else
  echo "==========================================================" | tee -a logs/pipeline.log
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Phase 2: Baselines (mask2former and hdnet scheduled separately)" | tee -a logs/pipeline.log
  echo "==========================================================" | tee -a logs/pipeline.log
  python scripts/run_experiments.py train-test --dataset wwtp --suite baselines --exclude mask2former hdnet --resume 2>&1 | tee -a logs/pipeline_baselines.log
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 2 Complete (Baselines)" | tee -a logs/pipeline.log
fi

echo "==========================================================" | tee -a logs/pipeline.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Phase 2.1: Mask2Former (COCO query budget, 30k schedule)" | tee -a logs/pipeline.log
echo "==========================================================" | tee -a logs/pipeline.log
python scripts/run_experiments.py train-test --dataset wwtp --models mask2former --resume 2>&1 | tee -a logs/pipeline_mask2former.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 2.1 Complete (mask2former)" | tee -a logs/pipeline.log

echo "==========================================================" | tee -a logs/pipeline.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Phase 2.2: HDNet" | tee -a logs/pipeline.log
echo "==========================================================" | tee -a logs/pipeline.log
python scripts/run_experiments.py train-test --dataset wwtp --models hdnet --resume 2>&1 | tee -a logs/pipeline_hdnet.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 2.2 Complete (HDNet)" | tee -a logs/pipeline.log

echo "==========================================================" | tee -a logs/pipeline.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Phase 2.4: RPGV v8" | tee -a logs/pipeline.log
echo "==========================================================" | tee -a logs/pipeline.log
python scripts/run_experiments.py train-test --dataset wwtp --models rpgv_v8 --resume 2>&1 | tee -a logs/pipeline_v8.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 2.4 Complete (RPGV v8)" | tee -a logs/pipeline.log

echo "==========================================================" | tee -a logs/pipeline.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Phase 2.5: Retraining RPGV v7 from scratch (Full 30k schedule, overwriting previous result)" | tee -a logs/pipeline.log
echo "==========================================================" | tee -a logs/pipeline.log
if [[ -d "work_dirs/wwtp/rpgv_v7" ]]; then
  echo "Existing rpgv_v7 will be cleared by --overwrite; no archive is kept." | tee -a logs/pipeline.log
fi
python scripts/run_experiments.py train-test --dataset wwtp --suite v7 --overwrite 2>&1 | tee -a logs/pipeline_v7_retrain_30k.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 2.5 Complete (RPGV v7 Fresh 30k)" | tee -a logs/pipeline.log

echo "==========================================================" | tee -a logs/pipeline.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Phase 3: V7 Ablations (Gated on V7 superiority)" | tee -a logs/pipeline.log
echo "==========================================================" | tee -a logs/pipeline.log
python scripts/run_experiments.py train-test --dataset wwtp --suite v7-ablations --resume 2>&1 | tee -a logs/pipeline_v7_ablations.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 3 Complete (V7 Ablations)" | tee -a logs/pipeline.log

echo "==========================================================" | tee -a logs/pipeline.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL PIPELINE PHASES FINISHED SUCCESSFULLY" | tee -a logs/pipeline.log
echo "==========================================================" | tee -a logs/pipeline.log
