"""Load OpenFilters abeles (Python fallback) for cross-check."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from paths import upstream_openfilters_root


def load_abeles(of_root: Path | None = None) -> Any:
    root = (of_root or upstream_openfilters_root()).resolve()
    for p in (str(root / "abeles"), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import abeles

    if not hasattr(abeles, "wvls"):
        raise RuntimeError("abeles Python backend failed to load")
    return abeles
