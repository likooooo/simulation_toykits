"""Load Freehand base optimization config from JSON (deploy-time tuning)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import filmstack_visualizer
from filmstack_simulation.presets import CUSTOM_PRESET_ID

_DEFAULT_JSON = Path(__file__).with_name("freehand_base_opt.json")
_COST_MODULE = Path(__file__).with_name("cost_freehand.py")


def resolve_opt_config_path() -> Path | None:
    env = os.environ.get("FREEHAND_OPT_CONFIG_PATH", "").strip()
    if env:
        path = Path(env)
        return path if path.is_file() else None
    return _DEFAULT_JSON if _DEFAULT_JSON.is_file() else None


_DEFAULT_N_WL = 80
_DEFAULT_THICKNESS_RANGE_PCT = 30.0
_VALID_COST_SCOPES = ("full", "zoom", "stroke")
_DEFAULT_COST_SCOPE = "full"


def get_freehand_cost_scope() -> str:
    """Return cost scope from base JSON: full, zoom, or stroke."""
    cfg = load_freehand_base_config()
    scope = str(cfg.get("freehand_cost_scope", _DEFAULT_COST_SCOPE)).lower()
    if scope not in _VALID_COST_SCOPES:
        raise ValueError(
            f"freehand_cost_scope must be one of {_VALID_COST_SCOPES}, got {scope!r}"
        )
    return scope


def get_freehand_n_wl() -> int:
    """Return wavelength grid point count from base JSON (deploy-time tuning)."""
    cfg = load_freehand_base_config()
    raw = cfg.get("n_wl", _DEFAULT_N_WL)
    n_wl = int(raw)
    if n_wl < 2:
        raise ValueError(f"n_wl must be >= 2, got {n_wl}")
    return n_wl


def get_freehand_initial_preset_id(valid_preset_ids: frozenset[str]) -> str:
    """Return initial preset id from base JSON (deploy-time tuning)."""
    cfg = load_freehand_base_config()
    preset_id = str(cfg.get("initial_preset_id") or CUSTOM_PRESET_ID)
    if preset_id not in valid_preset_ids:
        raise ValueError(
            f"initial_preset_id must be one of {sorted(valid_preset_ids)}, got {preset_id!r}"
        )
    return preset_id


def get_freehand_initial_formula() -> str:
    """Return initial filmstack formula from base JSON (deploy-time tuning)."""
    cfg = load_freehand_base_config()
    return str(cfg.get("initial_formula") or "")


def get_freehand_default_thickness_range_pct() -> float:
    """Return default per-layer thickness variation (%) from base JSON."""
    cfg = load_freehand_base_config()
    pct = float(cfg.get("default_thickness_range_pct", _DEFAULT_THICKNESS_RANGE_PCT))
    if not 0.0 <= pct <= 100.0:
        raise ValueError(f"default_thickness_range_pct must be in [0, 100], got {pct}")
    return pct


def load_freehand_base_config() -> dict[str, Any]:
    """Merge JSON base config with filmstack_visualizer defaults."""
    fv = filmstack_visualizer
    path = resolve_opt_config_path()
    base: dict[str, Any] = {}
    if path is not None:
        with path.open(encoding="utf-8") as f:
            base = json.load(f)
    cfg = fv.merge_filmstack_optimization_config(base)
    cost = cfg.get("cost_function") or {}
    cfg["cost_function"] = {
        "path": str(_COST_MODULE.resolve()),
        "name": str(cost.get("name") or "freehand_target"),
    }
    return cfg
