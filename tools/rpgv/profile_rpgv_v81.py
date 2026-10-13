#!/usr/bin/env python3
"""Synthetic v8.1 resource/gradient diagnostics, not an accuracy or causality test.

Reuses the v8 profiler with the v8.1 switches exposed, so the added content-aware
support, edge values and high-resolution local fusion can be priced individually.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.rpgv import profile_rpgv_v8 as v8_profile

CONFIG = 'configs/rpgv/v81/rpgv_v81.py'


def extra_arguments(parser):
    parser.add_argument('--no-edge-value', dest='use_edge_value', action='store_false')
    parser.add_argument('--no-content-support', dest='content_aware', action='store_false')
    parser.add_argument('--no-local-detail', dest='local_detail', action='store_false')


def main(argv=None):
    v8_profile.MODEL_TYPES = v8_profile.MODEL_TYPES + ('RPGVNetV81', 'RPGVRemoteSensingV81')
    return v8_profile.main(argv, default_config=CONFIG, extra_arguments=extra_arguments)


if __name__ == '__main__':
    raise SystemExit(main())
