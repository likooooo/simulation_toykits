#!/usr/bin/env python3
"""Generate DR comparison figures for all GranFilm Python ports."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from granfilm.paths import COMPARE_DIR, REPO_ROOT, SPHERE_COMPARE_DIR

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Match pytest runtime: chdir to artifacts so simulation imports resolve.
_artifacts = os.environ.get("SIMULATION_ARTIFACTS_DIR", str(REPO_ROOT / ".simulation_toolkits"))
os.environ.setdefault("SIMULATION_ARTIFACTS_DIR", _artifacts)
os.environ.setdefault("SIMULATION_DATABASE_DIR", str(Path(_artifacts) / "assets"))
os.chdir(_artifacts)
import simulation  # noqa: F401

from granfilm.build_granfilm_compare import (  # noqa: E402
    GRANFILM_COMPARE_SPECS,
    build_materials_db,
    prepare_toykits_runtime,
    process_case,
    resolve_granfilm_env,
)


def main() -> int:
    env_err = prepare_toykits_runtime()
    if env_err:
        print(f"Error: {env_err}", file=sys.stderr)
        return 1
    try:
        resolve_granfilm_env()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    db = build_materials_db()
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict[str, float] = {}
    for spec in GRANFILM_COMPARE_SPECS:
        out_dir = spec["out_root"] / spec["subdir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        row = process_case(spec, db, out_dir, write_legacy_viz=True)
        if row.get("status") == "ok":
            key = f"{spec['pipeline']}/{spec['subdir']}"
            val = row.get("max_abs_err", row.get("max_abs"))
            if isinstance(val, (int, float)):
                summary[key] = float(val)
            print(f"{key}: max|ΔDR|={float(val):.3e}")
        else:
            print(f"Skip {spec['id']}: {row.get('error', '')}")

    island_compare = SPHERE_COMPARE_DIR / "compare.png"
    if island_compare.is_file():
        shutil.copy2(island_compare, COMPARE_DIR / "compare.png")

    summary_path = COMPARE_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote figures under {COMPARE_DIR}")
    print(f"Summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
