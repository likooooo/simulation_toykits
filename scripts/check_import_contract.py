#!/usr/bin/env python3
"""Static import contract checks for portable packages."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

FORBIDDEN_TOP_LEVEL = frozenset({"core", "pages", "ui"})
PACKAGE_ROOTS = (
    Path(__file__).resolve().parent.parent / "simulation_database",
    Path(__file__).resolve().parent.parent / "filmstack_simulation",
    Path(__file__).resolve().parent.parent
    / "filmstack_simulation"
    / "filmstack_optimization",
)


def _top_level_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((node.lineno, alias.name.split(".")[0]))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.append((node.lineno, node.module.split(".")[0]))
    return found


def main() -> int:
    violations: list[str] = []
    for root in PACKAGE_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if path.name == "__init__.py":
                continue
            for lineno, top in _top_level_imports(path):
                if top in FORBIDDEN_TOP_LEVEL:
                    violations.append(
                        f"{path.relative_to(root.parent)}:{lineno}: forbidden import '{top}'"
                    )
    if violations:
        print("Import contract violations:", file=sys.stderr)
        for line in violations:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("Import contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
