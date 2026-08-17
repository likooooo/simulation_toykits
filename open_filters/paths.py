"""Path helpers for open_filters (cross-check + material export)."""

from __future__ import annotations

import os
from pathlib import Path


def package_root() -> Path:
    """``simulation_toykits/open_filters``."""
    return Path(__file__).resolve().parent


def toykits_root() -> Path:
    return package_root().parent


def crosscheck_dir() -> Path:
    return package_root() / "crosscheck"


def database_of_dir() -> Path:
    """OpenFilters materials tree under ``simulation_core/assets/database/of/``."""
    return simulation_database_root() / "of"


def simulation_database_root() -> Path:
    env = os.environ.get("SIMULATION_DATABASE_DIR", "").strip()
    if env:
        return Path(env).resolve()
    return toykits_root() / "simulation_core" / "assets" / "database"


def of_materials_release_dir() -> Path:
    """YAML materials consumed by ``simulation_database`` (``of/materials``)."""
    return simulation_database_root() / "of" / "materials"


def upstream_openfilters_root() -> Path:
    """OpenFilters source tree under GENERATE_GOLDEN_TOOLS_DIR."""
    env = os.environ.get("GENERATE_GOLDEN_TOOLS_DIR", "").strip()
    root = (
        Path(env).expanduser()
        if env
        else Path.home() / "repos" / "simulation_baseline_tools"
    )
    return (root / "OpenFilters").resolve()
