"""Step 4: island interaction and surface constitutive coefficients (Spheroid optics)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from granfilm.common.constants import PI
from granfilm.common.interaction import (
    effective_lattice_L,
    lattice_sum as _lattice_sum_core,
    renorm_polarizability,
    surf_const_quadrupole_values,
)
from granfilm.oblate_prolate.step0_init import SpheroidInitState


@dataclass
class SurfaceConstitutive:
    gamma: complex
    beta: complex
    tau: complex
    delta: complex


def lattice_sum(d_mu: float, n: int, state: SpheroidInitState, *, nint: int = 250) -> float:
    case = state.case
    return _lattice_sum_core(
        d_mu,
        n,
        network=case.network,
        R=state.R,
        Rapparent=state.Rapparent,
        density=state.density,
        lattice_const=case.lattice_const,
        levels=case.Levels,
        nint=nint,
    )


def surf_const_coef_quadrupole(
    alphad: np.ndarray,
    alphaq: np.ndarray,
    state: SpheroidInitState,
    ienergy: int,
    *,
    nint: int = 250,
) -> SurfaceConstitutive:
    """Spheroid optics_mod surf_const_coef_quadrupole (d = |div_surface|/R)."""
    case = state.case
    e1 = state.eps_vacuum
    e2 = state.eps_substrate[ienergy]
    d = abs(state.div_surface) / state.R
    density = state.density * state.R**2
    L = effective_lattice_L(
        network=case.network,
        lattice_const=case.lattice_const,
        R=state.R,
        density_dim=density,
    )
    gamma, beta, tau, delta = surf_const_quadrupole_values(
        alphad,
        alphaq,
        e1=e1,
        e2=e2,
        above=state.above,
        d=d,
        density=density,
        L=L,
        s2_mp=lattice_sum(0.0, 2, state, nint=nint),
        s2_imp=lattice_sum(d, 2, state, nint=nint),
        s3_mp=lattice_sum(0.0, 3, state, nint=nint),
        s3_imp=lattice_sum(d, 3, state, nint=nint),
        s4_mp=lattice_sum(0.0, 4, state, nint=nint),
        s4_imp=lattice_sum(d, 4, state, nint=nint),
    )
    return SurfaceConstitutive(gamma=gamma, beta=beta, tau=tau, delta=delta)


def step4_surface_coefficients(
    alpha: np.ndarray,
    state: SpheroidInitState,
    ienergy: int,
    *,
    alphaq: np.ndarray | None = None,
    nint: int = 250,
) -> SurfaceConstitutive:
    case = state.case
    interaction = case.interaction.strip().lower()
    if interaction == "none":
        return _surf_const_nointeract(alpha, state, ienergy)
    if interaction == "quadrupole":
        if alphaq is None:
            alphaq = np.zeros(3, dtype=np.complex128)
        return surf_const_coef_quadrupole(alpha, alphaq, state, ienergy, nint=nint)

    e1 = state.eps_vacuum
    d = abs(state.div_surface) / state.R
    density = state.density * state.R**2
    L = effective_lattice_L(
        network=case.network,
        lattice_const=case.lattice_const,
        R=state.R,
        density_dim=density,
    )

    s_mp = lattice_sum(0.0, 2, state, nint=nint)
    s_imp = lattice_sum(d, 2, state, nint=nint)
    if case.network.strip().upper() == "RPT":
        renorm_polarizability(
            d,
            alpha[0, :],
            eps_vacuum=e1,
            eps_substrate=state.eps_substrate[ienergy],
            coverage=state.coverage,
            R=state.R,
            Rapparent=state.Rapparent,
            levels=case.Levels,
            density=state.density,
            above=state.above,
            nint=nint,
        )
    sqr = np.sqrt(4.0 * PI / 5.0)

    paral = alpha[0, 0] / (4 * PI * e1)
    if state.above:
        e2 = state.eps_substrate[ienergy]
        eta = (e1 - e2) / (e1 + e2)
        factor_p = s_mp + eta * s_imp
        paral = 4 * PI * e1 * paral / (1.0 + paral * sqr / L**3 * factor_p)
        perp = alpha[0, 1] / (4 * PI * e1)
        factor_v = s_mp - eta * s_imp
        perp = 4 * PI * e1 * perp / (1.0 - 2.0 * perp * sqr / L**3 * factor_v)
        return SurfaceConstitutive(
            beta=density * perp / e1**2,
            gamma=density * paral,
            delta=density * d * (paral + perp) / e1,
            tau=density * d * paral,
        )

    e2 = state.eps_substrate[ienergy]
    eps = (e2 - e1) / (e2 + e1)
    factor_p = s_mp + eps * s_imp
    paral = alpha[0, 0] / (4 * PI * e2)
    paral = 4 * PI * e2 * paral / (1.0 + paral * sqr / L**3 * factor_p)
    perp = alpha[0, 1] / (4 * PI * e2)
    factor_v = s_mp - eps * s_imp
    perp = 4 * PI * e2 * perp / (1.0 - 2.0 * perp * sqr / L**3 * factor_v)
    return SurfaceConstitutive(
        beta=density * perp / e2**2,
        gamma=density * paral,
        delta=-density * d * (paral + perp) / e2,
        tau=-density * d * paral * e1 / e2,
    )


def _surf_const_yamaguchi_nointeract(
    alpha: np.ndarray,
    state: SpheroidInitState,
    ienergy: int,
) -> SurfaceConstitutive:
    """optics_mod surf_const_coef_nointeract as used by surf_const_coef_island_Yamaguchi."""
    e1 = state.eps_vacuum
    d = abs(state.div_surface) / state.R
    density = state.density * state.R**2
    if state.above:
        delta = -density / e1 * np.sum(alpha[1, :] - d * alpha[0, :])
        tau = -density * (alpha[1, 0] - d * alpha[0, 0])
        return SurfaceConstitutive(
            gamma=density * alpha[0, 0],
            beta=density * alpha[0, 1] / e1**2,
            delta=delta,
            tau=tau,
        )
    e2 = state.eps_substrate[ienergy]
    delta = -density / e2 * np.sum(alpha[1, :] + d * alpha[0, :])
    tau = -density * (e1 / e2) * (alpha[1, 0] + d * alpha[0, 0])
    return SurfaceConstitutive(
        gamma=density * alpha[0, 0],
        beta=density * alpha[0, 1] / e2**2,
        delta=delta,
        tau=tau,
    )


def step4_yamaguchi_surface_coefficients(
    alpha: np.ndarray,
    state: SpheroidInitState,
    ienergy: int,
    *,
    nint: int = 250,
) -> SurfaceConstitutive:
    """optics_mod surf_const_coef_island_Yamaguchi (none or dipole only)."""
    interaction = state.case.interaction.strip().lower()
    if interaction == "quadrupole":
        interaction = "dipole"
    if interaction == "none":
        return _surf_const_yamaguchi_nointeract(alpha, state, ienergy)
    if interaction == "dipole":
        return step4_surface_coefficients(alpha, state, ienergy, nint=nint)
    raise ValueError(f"Yamaguchi surface coefficients do not support interaction={interaction!r}")


def _surf_const_nointeract(alpha: np.ndarray, state: SpheroidInitState, ienergy: int) -> SurfaceConstitutive:
    e1 = state.eps_vacuum
    density = state.density * state.R**2
    if state.above:
        return SurfaceConstitutive(
            gamma=density * alpha[0, 0],
            beta=density * alpha[0, 1] / e1**2,
            delta=-density / e1 * alpha[0, 0],
            tau=-density * alpha[0, 1] / e1,
        )
    e2 = state.eps_substrate[ienergy]
    return SurfaceConstitutive(
        gamma=density * alpha[0, 0],
        beta=density * alpha[0, 1] / e2**2,
        delta=-density / e2 * alpha[0, 0],
        tau=-density * (e1 / e2) * alpha[0, 1] / e2,
    )
