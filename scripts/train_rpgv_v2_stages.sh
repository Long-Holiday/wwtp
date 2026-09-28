#!/usr/bin/env bash
# Compatibility entry point: v2 now uses a single joint run.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_rpgv_v2.sh" "$@"
