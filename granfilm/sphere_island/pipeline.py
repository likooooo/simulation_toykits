"""End-to-end GranFilm Sphere pipeline (all geometry types)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from granfilm.common.zeta import step1_zeta
from granfilm.sphere_island.case import GranFilmCase
from granfilm.sphere_island.step0_init import InitState, step0_init
from granfilm.sphere_island.step1_integrals import IntegralsVolume, step1_integrals
from granfilm.sphere_island.step2_system import step2_solve_multipoles, step2_system_matrix
from granfilm.sphere_island.step3_polarizability import step3_polarizabilities, step3_polarizabilities_quadrupole
from granfilm.sphere_island.step4_geometry import (
    coating_epsilon_grid,
    polarizabilities_cap,
    surf_const_coef_2film,
    surf_const_coef_cap,
    surf_const_coef_film,
)
from granfilm.sphere_island.step4_interaction import SurfaceConstitutive, step4_surface_coefficients
from granfilm.sphere_island.step5_fresnel import step5_fresnel


@dataclass
class GranFilmResult:
    energy: np.ndarray
    dr: np.ndarray
    alpha: np.ndarray  # (N, 2, 2)
    chi: list[SurfaceConstitutive]
    init: InitState
    zeta: np.ndarray | None = None
    integrals: IntegralsVolume | None = None
    integrals_tr1: IntegralsVolume | None = None
    mpoled: np.ndarray | None = None
    mid_energy_index: int = field(default=0)
    geometry: str = "island"


def _run_island(
    case: GranFilmCase,
    state: InitState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[SurfaceConstitutive], IntegralsVolume, IntegralsVolume, np.ndarray]:
    zeta = step1_zeta(case.Mpole_order, state.m_max)
    if state.above:
        int_tr1 = step1_integrals(1.0, case.MPpos, case.Mpole_order, state.m_max, nint=case.Nint)
        integrals = step1_integrals(case.tr, case.MPpos, case.Mpole_order, state.m_max, nint=case.Nint)
    else:
        # sphere.f90: tr -> -tr, MPpos -> -MPpos when param%above is false
        int_tr1 = step1_integrals(1.0, -case.MPpos, case.Mpole_order, state.m_max, nint=case.Nint)
        integrals = step1_integrals(-case.tr, -case.MPpos, case.Mpole_order, state.m_max, nint=case.Nint)
    n = len(state.energy)
    dr = np.empty(n, dtype=np.float64)
    alpha = np.empty((n, 2, 2), dtype=np.complex128)
    mpoled_all = np.empty((n, 2, 2), dtype=np.complex128)
    chi_list: list[SurfaceConstitutive] = []
    for i in range(n):
        mpo, mpq = step2_solve_multipoles(state, integrals, int_tr1, zeta, i)
        mpoled_all[i] = mpo
        alpha[i] = step3_polarizabilities(mpo, state, i)
        if case.interaction.strip().lower() == "quadrupole":
            if mpq is None:
                raise RuntimeError("mpoleq missing for quadrupole interaction")
            alphaq = step3_polarizabilities_quadrupole(mpq, state, i)
            chi = step4_surface_coefficients(alpha[i], state, i, alphaq=alphaq, nint=case.Nint)
        else:
            chi = step4_surface_coefficients(alpha[i], state, i, nint=case.Nint)
        chi_list.append(chi)
        dr[i] = step5_fresnel(chi, state, i)
    return dr, alpha, mpoled_all, chi_list, integrals, int_tr1, zeta


def _run_film(case: GranFilmCase, state: InitState) -> tuple[np.ndarray, list[SurfaceConstitutive]]:
    n = len(state.energy)
    dr = np.empty(n, dtype=np.float64)
    chi_list: list[SurfaceConstitutive] = []
    for i in range(n):
        chi = surf_const_coef_film(state, i)
        chi_list.append(chi)
        dr[i] = step5_fresnel(chi, state, i)
    return dr, chi_list


def _run_2film(
    case: GranFilmCase,
    state: InitState,
    materials_db: Mapping[str, Any],
) -> tuple[np.ndarray, list[SurfaceConstitutive]]:
    eps_coating = coating_epsilon_grid(state, materials_db)
    n = len(state.energy)
    dr = np.empty(n, dtype=np.float64)
    chi_list: list[SurfaceConstitutive] = []
    for i in range(n):
        chi = surf_const_coef_2film(state, i, eps_coating)
        chi_list.append(chi)
        dr[i] = step5_fresnel(chi, state, i)
    return dr, chi_list


def _run_thin_cap(case: GranFilmCase, state: InitState) -> tuple[np.ndarray, np.ndarray, list[SurfaceConstitutive]]:
    n = len(state.energy)
    dr = np.empty(n, dtype=np.float64)
    alpha = np.empty((n, 2, 2), dtype=np.complex128)
    chi_list: list[SurfaceConstitutive] = []
    for i in range(n):
        alpha[i] = polarizabilities_cap(state, i)
        chi = surf_const_coef_cap(alpha[i], state, i, nint=case.Nint)
        chi_list.append(chi)
        dr[i] = step5_fresnel(chi, state, i)
    return dr, alpha, chi_list


def run_granfilm_sphere(
    case: GranFilmCase,
    materials_db: Mapping[str, Any],
    *,
    viz_dir: Path | None = None,
    write_viz: bool = True,
    baseline: Any | None = None,
) -> GranFilmResult:
    state = step0_init(case, materials_db)
    geom = case.geometry.strip().lower()
    n = len(state.energy)
    mid = n // 2

    if geom == "island":
        dr, alpha, mpoled, chi, integrals, int_tr1, zeta = _run_island(case, state)
        result = GranFilmResult(
            energy=state.energy,
            dr=dr,
            alpha=alpha,
            chi=chi,
            init=state,
            zeta=zeta,
            integrals=integrals,
            integrals_tr1=int_tr1,
            mpoled=mpoled,
            mid_energy_index=mid,
            geometry=geom,
        )
    elif geom == "film":
        dr, chi = _run_film(case, state)
        result = GranFilmResult(
            energy=state.energy,
            dr=dr,
            alpha=np.zeros((n, 2, 2), dtype=np.complex128),
            chi=chi,
            init=state,
            mid_energy_index=mid,
            geometry=geom,
        )
    elif geom == "2film":
        dr, chi = _run_2film(case, state, materials_db)
        result = GranFilmResult(
            energy=state.energy,
            dr=dr,
            alpha=np.zeros((n, 2, 2), dtype=np.complex128),
            chi=chi,
            init=state,
            mid_energy_index=mid,
            geometry=geom,
        )
    elif geom == "thin_cap":
        dr, alpha, chi = _run_thin_cap(case, state)
        result = GranFilmResult(
            energy=state.energy,
            dr=dr,
            alpha=alpha,
            chi=chi,
            init=state,
            mid_energy_index=mid,
            geometry=geom,
        )
    else:
        raise ValueError(f"unsupported geometry: {case.geometry!r}")

    if viz_dir is not None and write_viz:
        from granfilm.sphere_island import viz

        matrix_fn = None
        if geom == "island" and result.integrals is not None and result.integrals_tr1 is not None and result.zeta is not None:
            matrix_fn = lambda: step2_system_matrix(
                state, result.integrals, result.integrals_tr1, result.zeta, mid, m=0
            )
        viz.write_all_step_figures(result, viz_dir, matrix_fn=matrix_fn, baseline=baseline)

    return result
