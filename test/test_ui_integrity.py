"""Static integrity checks for simulation_toykits host UI modules."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]

_TOOLKIT_PAGE_DIRS = (
    "filmstack_toolkits",
    "gaussian_optics_toolkits",
    "simulation_toykits",
)


def _ui_python_files() -> list[Path]:
    candidates = [
        _REPO / "app.py",
        _REPO / "common.py",
        *(_REPO / "pages").rglob("*.py"),
        *(_REPO / "core").rglob("*.py"),
        *(_REPO / "ui").rglob("*.py"),
    ]
    return sorted(p for p in candidates if p.is_file())


@pytest.mark.parametrize(
    "path",
    _ui_python_files(),
    ids=lambda p: str(p.relative_to(_REPO)),
)
def test_ui_modules_parse(path: Path) -> None:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_app_py_entrypoint_complete() -> None:
    text = (_REPO / "app.py").read_text(encoding="utf-8")
    assert "st.navigation" in text
    assert "pg.run()" in text
    assert "get_simulation_database" in text
    assert (_REPO / "pages" / "main.py").is_file()
    for subdir in _TOOLKIT_PAGE_DIRS:
        pages = list((_REPO / "pages" / subdir).glob("*.py"))
        assert pages, f"missing pages under pages/{subdir}/"
