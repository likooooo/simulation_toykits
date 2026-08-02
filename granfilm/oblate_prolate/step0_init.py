"""Step 0: derived spheroid parameters and ε grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from granfilm.common.constants import EPS_VACUUM, PI
from granfilm.common.materials import epsilon_grid
from granfilm.oblate_prolate.case import SpheroidCase


@dataclass
class SpheroidInitState:
    case: SpheroidCase
    energy: np.ndarray
    eps_island: np.ndarray
    eps_substrate: np.ndarray
    eps_coating: np.ndarray | None
    eps_vacuum: float
    density: float
    Rapparent: float
    R: float
    coverage: float
    volume: float
    SR: float
    xi0: float
    a: float
    island_type: str
    above: bool
    div_surface: float
    theta0_calc: float
    phi0_calc: float
    lattice_const_eff: float


def _derived_geometry(case: SpheroidCase) -> dict[str, float | bool | str]:
    tr_raw = case.tr
    if tr_raw >= 0.0:
        rapp = case.R_par
        above = True
        tr = tr_raw
    else:
        rapp = case.R_par * np.sqrt(1.0 - tr_raw**2)
        above = False
        tr = abs(tr_raw)

    network = case.network.strip().lower()
    if network == "square":
        density = 1.0 / case.lattice_const**2
        lattice_eff = case.lattice_const
    elif network == "hexagonal":
        density = 1.0 / (np.sqrt(3.0) / 2.0 * case.lattice_const**2)
        lattice_eff = case.lattice_const
    elif network in {"mft", "rpt"}:
        density = case.coverage / (PI * rapp**2)
        lattice_eff = 1.0 / np.sqrt(density)
    else:
        raise ValueError(f"unsupported network: {case.network}")

    coverage = density * PI * rapp**2

    if case.R_par > case.R_per:
        island_type = "oblate"
        xi0 = case.R_per / np.sqrt(case.R_par**2 - case.R_per**2)
        a = case.R_per / xi0
    elif case.R_par < case.R_per:
        island_type = "prolate"
        xi0 = case.R_per / np.sqrt(case.R_per**2 - case.R_par**2)
        a = case.R_per / xi0
    else:
        raise ValueError("sphere geometry is not supported in spheroid model")

    volume = PI * a**3 * xi0 * (2.0 / 3.0 + tr - tr**3 / 3.0)
    if island_type == "oblate":
        volume *= xi0**2 + 1.0
    else:
        volume *= xi0**2 - 1.0

    sr = 2.0 * rapp / (case.R_per * (1.0 + tr))
    div_surface = tr * case.R_per
    r_scale = case.R_per

    return {
        "density": density,
        "Rapparent": rapp,
        "coverage": coverage,
        "volume": volume,
        "SR": sr,
        "xi0": xi0,
        "a": a,
        "island_type": island_type,
        "above": above,
        "div_surface": div_surface,
        "tr": tr,
        "R": r_scale,
        "lattice_const_eff": lattice_eff,
    }


def step0_init(case: SpheroidCase, materials_db: Mapping[str, Any]) -> SpheroidInitState:
    energy = np.linspace(case.energy_min, case.energy_max, case.Nenergy, dtype=np.float64)
    geom = _derived_geometry(case)
    tr_eff = float(geom["tr"])
    eps_island = epsilon_grid(
        materials_db,
        case.island,
        energy,
        geometry=case.geometry,
        tr=tr_eff,
        R_nm=case.R_per,
        R_par_nm=case.R_par if str(geom["island_type"]) == "prolate" else None,
        mean_free_path=case.mean_free_path,
        surface_effects=case.surface_effects,
        temperature_k=case.temperature,
    )
    eps_substrate = epsilon_grid(
        materials_db,
        case.substrate,
        energy,
        mean_free_path="none",
    )
    eps_coating = None
    if case.geometry.strip().lower() == "coated":
        eps_coating = epsilon_grid(
            materials_db,
            case.coating,
            energy,
            mean_free_path="none",
        )
    return SpheroidInitState(
        case=case,
        energy=energy,
        eps_island=eps_island,
        eps_substrate=eps_substrate,
        eps_coating=eps_coating,
        eps_vacuum=EPS_VACUUM,
        density=float(geom["density"]),
        Rapparent=float(geom["Rapparent"]),
        R=float(geom["R"]),
        coverage=float(geom["coverage"]),
        volume=float(geom["volume"]),
        SR=float(geom["SR"]),
        xi0=float(geom["xi0"]),
        a=float(geom["a"]),
        island_type=str(geom["island_type"]),
        above=bool(geom["above"]),
        div_surface=float(geom["div_surface"]),
        theta0_calc=case.theta0,
        phi0_calc=0.0,
        lattice_const_eff=float(geom["lattice_const_eff"]),
    )
