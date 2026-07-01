"""Bootstrap simulation from simulation_toykits (.simulation_toolkits runtime)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from paths import toykits_root, upstream_openfilters_root


def openfilters_upstream_root():
    """OpenFilters source repo (abeles); alias kept for callers."""
    return upstream_openfilters_root()


def bootstrap_toykits_session() -> Path:
    """Set env vars and sys.path; import simulation. Returns toykits root."""
    tk = toykits_root()
    artifacts = tk / ".simulation_toolkits"
    database = tk / "simulation_core" / "assets" / "database"
    plugins = tk / "simulation_core" / "simulation_plugins"
    cc = tk / "open_filters" / "crosscheck"

    os.environ.setdefault("SIMULATION_ARTIFACTS_DIR", str(artifacts))
    os.environ.setdefault("SIMULATION_DATABASE_DIR", str(database))

    for p in (str(tk / "open_filters"), str(cc), str(plugins), str(artifacts)):
        if p not in sys.path:
            sys.path.insert(0, p)

    import simulation  # noqa: F401

    return tk


def simulation_available() -> bool:
    tk = toykits_root()
    so = tk / ".simulation_toolkits" / "simulation.so"
    return so.is_file() or any(tk.glob(".simulation_toolkits/simulation*.so"))
