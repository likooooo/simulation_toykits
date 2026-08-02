"""Physical and numerical constants matching GranFilm global_def / initialize_mod."""

from __future__ import annotations

import math
import os
from pathlib import Path

PI = math.pi
HBAR_EV_S = 6.582122e-16  # eV·s
C_NM_S = 2.99792458e17  # nm/s
EV_TO_UM = 1.23984193  # hc in eV·µm
EPS_VACUUM = 1.0


def _tools_dir() -> Path:
    env = os.environ.get("GENERATE_GOLDEN_TOOLS_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / "repos").resolve()


DEFAULT_GRANFILM_DIR = str((_tools_dir() / "GranFilm-v1.0").resolve())
