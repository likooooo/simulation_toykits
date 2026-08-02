"""Dielectric constants from GranFilm gf database (SOPRA .nk + finite_size)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from granfilm.common.sopra_dielectric import (
    FiniteSizeParams,
    epsilon_from_gf_material,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
GF_DB_DIR = _REPO_ROOT / "simulation_core" / "assets" / "database" / "gf"


def build_granfilm_materials_db(sim_db: Any | None = None) -> dict[str, Any]:
    import importlib.util

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("toykits_common", root / "common.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load toykits common.py from {root}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    keys = [
        ["gf", "materials", "ag.yml"],
        ["gf", "materials", "mgo.yml"],
        ["gf", "materials", "ti.yml"],
        ["gf", "materials", "tio2t.yml"],
    ]
    return mod.build_materials_db_from_path_keys(keys, sim_db=sim_db)


def load_finite_size_params(material_name: str) -> FiniteSizeParams | None:
    path = GF_DB_DIR / "finite_size" / f"{material_name}.yml"
    if not path.is_file():
        return None
    return FiniteSizeParams.from_yaml(path)


def epsilon_grid(
    materials_db: Mapping[str, Any],
    name: str,
    energy_ev: np.ndarray,
    *,
    geometry: str = "island",
    tr: float = 0.0,
    R_nm: float = 5.0,
    R_par_nm: float | None = None,
    mean_free_path: str = "none",
    surface_effects: bool = False,
    temperature_k: float = 300.0,
    A: float = 0.8,
    inv_tau_eV: float | None = None,
    manual_percent: float | None = None,
) -> np.ndarray:
    """ε(ω) via Fortran dielectric_constants (+ optional finite_size correction)."""
    mat = materials_db[name]
    fs = load_finite_size_params(name) if mean_free_path.strip().lower() != "none" else None
    return epsilon_from_gf_material(
        mat,
        np.asarray(energy_ev, dtype=np.float64),
        finite_size=fs,
        geometry=geometry,
        tr=tr,
        R_nm=R_nm,
        R_par_nm=R_par_nm,
        mean_free_path=mean_free_path,
        surface_effects=surface_effects,
        temperature_k=temperature_k,
        A=A,
        inv_tau_eV=inv_tau_eV,
        manual_percent=manual_percent,
    )
