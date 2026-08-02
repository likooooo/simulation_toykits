"""GranFilm package paths: all local test scripts and generated artifacts live here."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

OUTPUT_DIR = PACKAGE_DIR / "output"
COMPARE_DIR = OUTPUT_DIR / "compare"
SPHERE_COMPARE_DIR = COMPARE_DIR / "sphere_island"
OBLATE_COMPARE_DIR = COMPARE_DIR / "oblate_prolate"
LOG_DIR = OUTPUT_DIR / "logs"
BASELINE_NPZ = OUTPUT_DIR / "baseline_dr.npz"
SPHEROID_BASELINE_NPZ = OUTPUT_DIR / "baseline_spheroid_dr.npz"
BISECT_LOG = LOG_DIR / "bisect_align.ndjson"
