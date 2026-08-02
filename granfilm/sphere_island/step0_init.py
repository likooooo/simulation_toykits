"""Step 0: derived parameters and ε grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from granfilm.sphere_island.case import GranFilmCase
from granfilm.common.constants import EPS_VACUUM, PI
from granfilm.common.materials import epsilon_grid


@dataclass
class InitState:
    case: GranFilmCase
    energy: np.ndarray
    eps_island: np.ndarray
    eps_substrate: np.ndarray
    eps_vacuum: float
    density: float
    Rapparent: float
    coverage: float
    volume: float
    SR: float
    above: bool
    m_max: int
    theta0_calc: float
    phi0_calc: float
    div_surface: float
    lattice_const_eff: float


def _derived_geometry(case: GranFilmCase) -> dict[str, float | bool | int]:
    if case.tr >= 0.0:
        rapp = case.R
    else:
        rapp = case.R * np.sqrt(1.0 - case.tr**2)

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
    volume = PI * case.R**3 * (2.0 / 3.0 + case.tr - case.tr**3 / 3.0)
    sr = 2.0 * rapp / (1.0 + case.tr) / case.R
    above = case.tr >= case.MPpos

    interaction = case.interaction.strip().lower()
    if interaction in {"none", "dipole"}:
        m_max = 1
    elif interaction == "quadrupole":
        m_max = 2
    else:
        raise ValueError(f"unsupported interaction: {case.interaction}")

    return {
        "density": density,
        "Rapparent": rapp,
        "coverage": coverage,
        "volume": volume,
        "SR": sr,
        "above": above,
        "m_max": m_max,
        "lattice_const_eff": lattice_eff,
    }


def step0_init(case: GranFilmCase, materials_db: Mapping[str, Any]) -> InitState:
    energy = np.linspace(case.energy_min, case.energy_max, case.Nenergy, dtype=np.float64)
    geom = case.geometry.strip().lower()
    if geom in {"film", "2film"}:
        r_nm = case.film_thickness1
        tr_eff = 0.0
    else:
        r_nm = case.R
        tr_eff = case.tr
    eps_island = epsilon_grid(
        materials_db,
        case.island,
        energy,
        geometry=case.geometry,
        tr=tr_eff,
        R_nm=r_nm,
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
    geom = _derived_geometry(case)
    return InitState(
        case=case,
        energy=energy,
        eps_island=eps_island,
        eps_substrate=eps_substrate,
        eps_vacuum=EPS_VACUUM,
        density=float(geom["density"]),
        Rapparent=float(geom["Rapparent"]),
        coverage=float(geom["coverage"]),
        volume=float(geom["volume"]),
        SR=float(geom["SR"]),
        above=bool(geom["above"]),
        m_max=int(geom["m_max"]),
        theta0_calc=case.theta0,
        phi0_calc=0.0,
        div_surface=case.tr * case.R,
        lattice_const_eff=float(geom["lattice_const_eff"]),
    )
