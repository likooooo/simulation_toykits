"""Surface constitutive coefficients for film / 2film / thin_cap geometries."""

from __future__ import annotations

import numpy as np

from granfilm.common.constants import PI
from granfilm.common.materials import epsilon_grid
from granfilm.sphere_island.step0_init import InitState
from granfilm.sphere_island.step4_interaction import SurfaceConstitutive


def surf_const_coef_film(state: InitState, ienergy: int) -> SurfaceConstitutive:
    """Vlieger/Bedeaux thin continuous film (optics_mod surf_const_coef_film)."""
    e1 = state.eps_vacuum
    e3 = state.eps_island[ienergy]
    t = state.case.film_thickness1
    return SurfaceConstitutive(
        gamma=(e3 - e1) * t,
        beta=(1.0 / e1 - 1.0 / e3) * t,
        tau=t**2 * (e3 - e1) / 2,
        delta=t**2 * (e3 / e1 - e1 / e3) / 2,
    )


def surf_const_coef_2film(state: InitState, ienergy: int, eps_coating: np.ndarray) -> SurfaceConstitutive:
    """Stacking of two thin films (optics_mod surf_const_coef_2film)."""
    e1 = state.eps_vacuum
    t1 = state.case.film_thickness1
    t2 = state.case.film_thickness2
    e3_1 = state.eps_island[ienergy]
    e3_2 = eps_coating[ienergy]

    scc1 = SurfaceConstitutive(
        gamma=(e3_1 - e1) * t1,
        beta=(1.0 / e1 - 1.0 / e3_1) * t1,
        tau=t1**2 * (e3_1 - e1) / 2,
        delta=t1**2 * (e3_1 / e1 - e1 / e3_1) / 2,
    )
    scc2 = SurfaceConstitutive(
        gamma=(e3_2 - e1) * t2,
        beta=(1.0 / e1 - 1.0 / e3_2) * t2,
        tau=t2**2 * (e3_2 - e1) / 2,
        delta=t2**2 * (e3_2 / e1 - e1 / e3_2) / 2,
    )
    return SurfaceConstitutive(
        gamma=scc1.gamma + scc2.gamma,
        beta=scc1.beta + scc2.beta,
        tau=scc1.tau + scc2.tau,
        delta=scc1.delta + scc2.delta + (scc2.gamma * scc1.beta - scc1.gamma * scc2.beta) / 2,
    )


def polarizabilities_cap(state: InitState, ienergy: int) -> np.ndarray:
    """Thin spherical cap limited development (optics_mod polarizabilities_cap)."""
    e1 = state.eps_vacuum
    e2 = state.eps_substrate[ienergy]
    e3 = state.eps_island[ienergy]
    h = 1.0 - abs(state.case.tr)
    volume = PI * h**2 * (1.0 - h / 3.0)
    alpha = np.zeros((2, 2), dtype=np.complex128)
    alpha[0, 0] = volume * (e3 - e1)
    alpha[0, 1] = volume * e2**2 * (e3 - e1) / (e1 * e3)
    alpha[1, 0] = -volume * (e3 - e1)
    alpha[1, 1] = -volume * e2**2 * (e3 - e1) / (e1 * e3)
    return alpha


def surf_const_coef_cap(alpha: np.ndarray, state: InitState, ienergy: int, *, nint: int = 250) -> SurfaceConstitutive:
    """
    After polarizabilities_cap Fortran sets tr=1, MPpos=0 (d=1) before surf_const_coef_*.
    param%above is fixed at initialize from the original tr/MPpos (not updated in cap).
    """
    case = state.case
    interaction = case.interaction.strip().lower()
    e1 = state.eps_vacuum
    e2 = state.eps_substrate[ienergy]
    d = 1.0
    density = state.density * case.R**2
    if interaction == "none":
        if state.above:
            return SurfaceConstitutive(
                gamma=density * alpha[0, 0],
                beta=density * alpha[0, 1] / e1**2,
                delta=-density / e1 * np.sum(alpha[1, :] - d * alpha[0, :]),
                tau=-density * (alpha[1, 0] - d * alpha[0, 0]),
            )
        return SurfaceConstitutive(
            gamma=density * alpha[0, 0],
            beta=density * alpha[0, 1] / e2**2,
            delta=-density / e2 * np.sum(alpha[1, :] + d * alpha[0, :]),
            tau=-density * (e1 / e2) * (alpha[1, 0] + d * alpha[0, 0]),
        )
    if interaction == "dipole":
        from granfilm.common.interaction import effective_lattice_L
        from granfilm.sphere_island.step4_interaction import lattice_sum

        L = effective_lattice_L(
            network=case.network,
            lattice_const=case.lattice_const,
            R=case.R,
            density_dim=density,
        )
        sqr = np.sqrt(4.0 * PI / 5.0)
        s_mp = lattice_sum(0.0, 2, state, nint=nint)
        s_imp = lattice_sum(d, 2, state, nint=nint)
        if state.above:
            eta = (e1 - e2) / (e1 + e2)
            paral = alpha[0, 0] / (4 * PI * e1)
            factor = s_mp + eta * s_imp
            paral = 4 * PI * e1 * paral / (1.0 + paral * sqr / L**3 * factor)
            perp = alpha[0, 1] / (4 * PI * e1)
            factor = s_mp - eta * s_imp
            perp = 4 * PI * e1 * perp / (1.0 - 2.0 * perp * sqr / L**3 * factor)
            return SurfaceConstitutive(
                beta=density * perp / e1**2,
                gamma=density * paral,
                delta=density * d * (paral + perp) / e1,
                tau=density * d * paral,
            )
        eta = (e2 - e1) / (e2 + e1)
        paral = alpha[0, 0] / (4 * PI * e2)
        factor = s_mp + eta * s_imp
        paral = 4 * PI * e2 * paral / (1.0 + paral * sqr / L**3 * factor)
        perp = alpha[0, 1] / (4 * PI * e2)
        factor = s_mp - eta * s_imp
        perp = 4 * PI * e2 * perp / (1.0 - 2.0 * perp * sqr / L**3 * factor)
        return SurfaceConstitutive(
            beta=density * perp / e2**2,
            gamma=density * paral,
            delta=-density * d * (paral + perp) / e2,
            tau=-density * d * paral * e1 / e2,
        )
    raise ValueError(f"thin_cap interaction {interaction!r} not supported")


def coating_epsilon_grid(state: InitState, materials_db) -> np.ndarray:
    """Coating layer ε with finite_size correction using film_thickness2 as R_eff."""
    case = state.case
    return epsilon_grid(
        materials_db,
        case.coating,
        state.energy,
        geometry="2film",
        tr=0.0,
        R_nm=case.film_thickness2,
        mean_free_path=case.mean_free_path,
        surface_effects=case.surface_effects,
        temperature_k=case.temperature,
    )
