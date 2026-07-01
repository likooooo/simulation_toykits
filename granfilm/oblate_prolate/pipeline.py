"""End-to-end GranFilm oblate/prolate spheroid pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from granfilm.common.zeta import step1_zeta
from granfilm.oblate_prolate.case import SpheroidCase
from granfilm.oblate_prolate.step0_init import SpheroidInitState, step0_init
from granfilm.oblate_prolate.step1_integrals import SpheroidIntegrals, step1_integrals, step1_xz_at_xi0
from granfilm.oblate_prolate.step2_system import step2_solve_multipoles
from granfilm.oblate_prolate.step3_polarizability import step3_polarizabilities
from granfilm.oblate_prolate.step4_interaction import (
    SurfaceConstitutive,
    step4_surface_coefficients,
    step4_yamaguchi_surface_coefficients,
)
from granfilm.oblate_prolate.step5_fresnel import step5_fresnel
from granfilm.oblate_prolate.yamaguchi import yamaguchi_polarizabilities


@dataclass
class SpheroidResult:
    energy: np.ndarray
    dr: np.ndarray
    alpha: np.ndarray
    chi: list[SurfaceConstitutive]
    init: SpheroidInitState
    zeta: np.ndarray
    integrals: SpheroidIntegrals
    mpoled: np.ndarray


def run_granfilm_spheroid(
    case: SpheroidCase,
    materials_db: Mapping[str, Any],
    *,
    viz_dir: Path | None = None,
    write_viz: bool = True,
    baseline: Any | None = None,
) -> SpheroidResult:
    state = step0_init(case, materials_db)
    n = len(state.energy)
    dr = np.empty(n, dtype=np.float64)
    alpha = np.empty((n, 2, 2), dtype=np.complex128)
    mpoled_all = np.empty((n, 2, 2), dtype=np.complex128)
    chi_list: list[SurfaceConstitutive] = []

    geom = case.geometry.strip().lower()
    if geom in {"yamaguchi", "coated"}:
        alpha[:] = yamaguchi_polarizabilities(state, nint=case.Nint)
        mpoled_all[:] = alpha
        zeta = step1_zeta(1, m_max=1)
        integrals = step1_integrals(state, nint=case.Nint)
        for i in range(n):
            chi = step4_yamaguchi_surface_coefficients(alpha[i], state, i, nint=case.Nint)
            chi_list.append(chi)
            dr[i] = step5_fresnel(chi, state, i)
    else:
        zeta = step1_zeta(case.Mpole_order, m_max=1)
        integrals = step1_integrals(state, nint=case.Nint)
        xz = step1_xz_at_xi0(state)
        for i in range(n):
            mpo = step2_solve_multipoles(state, integrals, xz, zeta, i)
            mpoled_all[i] = mpo
            alpha[i] = step3_polarizabilities(mpo, state, i)
            chi = step4_surface_coefficients(alpha[i], state, i, nint=case.Nint)
            chi_list.append(chi)
            dr[i] = step5_fresnel(chi, state, i)

    result = SpheroidResult(
        energy=state.energy,
        dr=dr,
        alpha=alpha,
        chi=chi_list,
        init=state,
        zeta=zeta,
        integrals=integrals,
        mpoled=mpoled_all,
    )

    if viz_dir is not None and write_viz:
        from granfilm.oblate_prolate import viz

        viz.write_all_figures(result, viz_dir, baseline=baseline)

    return result
