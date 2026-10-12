#!/usr/bin/env bash
# Sequential pipeline for WWTP 1024 experiments:
# Phase 1: RPGV v7 train and test evaluation
# Phase 2: Baselines train and test evaluation
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

echo "==========================================================" | tee -a logs/pipeline.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Phase 1: RPGV v7" | tee -a logs/pipeline.log
echo "==========================================================" | tee -a logs/pipeline.log
python scripts/run_experiments.py train-test --dataset wwtp --suite v7 --resume 2>&1 | tee -a logs/pipeline_v7.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 1 Complete (RPGV v7)" | tee -a logs/pipeline.log

echo "==========================================================" | tee -a logs/pipeline.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Phase 2: Baselines" | tee -a logs/pipeline.log
echo "==========================================================" | tee -a logs/pipeline.log
python scripts/run_experiments.py train-test --dataset wwtp --suite baselines --resume 2>&1 | tee -a logs/pipeline_baselines.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 2 Complete (Baselines)" | tee -a logs/pipeline.log

echo "==========================================================" | tee -a logs/pipeline.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Phase 2.4: RPGV v8" | tee -a logs/pipeline.log
echo "==========================================================" | tee -a logs/pipeline.log
python scripts/run_experiments.py train-test --dataset wwtp --models rpgv_v8 --resume 2>&1 | tee -a logs/pipeline_v8.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 2.4 Complete (RPGV v8)" | tee -a logs/pipeline.log

echo "==========================================================" | tee -a logs/pipeline.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Phase 2.5: Retraining RPGV v7 from scratch (Full 30k schedule, overwriting previous result)" | tee -a logs/pipeline.log
echo "==========================================================" | tee -a logs/pipeline.log
if [[ -d "work_dirs/wwtp/rpgv_v7" ]]; then
  echo "Archiving previous rpgv_v7 to rpgv_v7_resumed_archive..." | tee -a logs/pipeline.log
  rm -rf work_dirs/wwtp/rpgv_v7_resumed_archive
  cp -r work_dirs/wwtp/rpgv_v7 work_dirs/wwtp/rpgv_v7_resumed_archive
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
