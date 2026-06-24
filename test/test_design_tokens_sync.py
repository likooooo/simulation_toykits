"""design_tokens.css copies in self-contained packages must stay identical."""

from __future__ import annotations

import hashlib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_TOKEN_PATHS = (
    _REPO / "ui" / "design_tokens.css",
    _REPO / "filmstack_simulation" / "design_tokens.css",
    _REPO / "simulation_database" / "design_tokens.css",
    _REPO / "filmstack_simulation" / "filmstack_optimization" / "design_tokens.css",
)


def _file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_design_tokens_css_copies_match() -> None:
    digests = [_file_md5(p) for p in _TOKEN_PATHS]
    assert len(set(digests)) == 1, (
        "design_tokens.css drifted between ui/, filmstack_simulation/, simulation_database/, filmstack_simulation/filmstack_optimization/"
    )
