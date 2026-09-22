#!/usr/bin/env bash
# Start the complete WWTP gallery only after the current training container exits.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <training-container-id>" >&2
  exit 2
fi

cd "$(dirname "$0")/.."
echo "Waiting for training container $1 to exit before starting inference"
docker wait "$1"
echo "Training container exited; starting WWTP val/test inference"
docker compose run --rm wwtp python tools/visualize_wwtp_predictions.py run
echo "WWTP gallery complete"
