# -*- coding: utf-8 -*-
"""
Sprint 01 – US-03

Goal: run the pre-processing pipeline that creates lag features, return-based
features and the future targets before saving the processed CSV. This script is
kept simple so it can be demoed in class (no CLIs or env vars needed).
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow "python scripts/Sprint...py" to import the src package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data.pipeline import PipelineConfig, run_pipeline, summarize_artifacts


def main() -> None:
    config = PipelineConfig()
    artifacts = run_pipeline(config)

    print("=== Sprint 01 US-03: Pre-processing report ===")
    for label, value in summarize_artifacts(artifacts).items():
        print(f"{label.capitalize()}: {value}")


if __name__ == "__main__":
    main()
