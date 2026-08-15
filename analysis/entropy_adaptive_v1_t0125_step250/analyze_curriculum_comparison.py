#!/usr/bin/env python3
"""Run the shared adaptive-v1 curriculum analysis with this experiment's defaults."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "analysis") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "analysis"))

from plot_adaptive_v1_curriculum import main


if __name__ == "__main__":
    main()
