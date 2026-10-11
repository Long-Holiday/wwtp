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
export PYTHONUNBUFFERED=1

mkdir -p logs

echo "==========================================================" | tee -a logs/pipeline.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Phase 1: RPGV v7" | tee -a logs/pipeline.log
echo "==========================================================" | tee -a logs/pipeline.log
python scripts/run_experiments.py train-test --dataset wwtp --suite v7 2>&1 | tee -a logs/pipeline_v7.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 1 Complete (RPGV v7)" | tee -a logs/pipeline.log

echo "==========================================================" | tee -a logs/pipeline.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Phase 2: Baselines" | tee -a logs/pipeline.log
echo "==========================================================" | tee -a logs/pipeline.log
python scripts/run_experiments.py train-test --dataset wwtp --suite baselines 2>&1 | tee -a logs/pipeline_baselines.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 2 Complete (Baselines)" | tee -a logs/pipeline.log

echo "==========================================================" | tee -a logs/pipeline.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Phase 3: V7 Ablations" | tee -a logs/pipeline.log
echo "==========================================================" | tee -a logs/pipeline.log
python scripts/run_experiments.py train-test --dataset wwtp --suite v7-ablations 2>&1 | tee -a logs/pipeline_v7_ablations.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 3 Complete (V7 Ablations)" | tee -a logs/pipeline.log

echo "==========================================================" | tee -a logs/pipeline.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL PIPELINE PHASES FINISHED SUCCESSFULLY" | tee -a logs/pipeline.log
echo "==========================================================" | tee -a logs/pipeline.log
