#!/usr/bin/env bash
set -euo pipefail

python scripts/run_rpgv_v2_ablations.py run --variants progressive_rgr progressive_contour --seeds 42 --max-iters 100000 --work-root work_dirs/rpgv_v2_quick_screen/official_confirmation
