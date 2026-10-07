#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "${PYTHON_BIN:-python}" scripts/run_experiments.py train --dataset "${DATASET:-wwtp}" "$@"
