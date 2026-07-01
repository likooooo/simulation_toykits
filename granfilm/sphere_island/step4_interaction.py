"""Step 4: island interaction and surface constitutive coefficients."""

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
from granfilm.sphere_island.step0_init import InitState


@dataclass
class SurfaceConstitutive:
    gamma: complex
    beta: complex
    tau: complex
    delta: complex


def lattice_sum(d_mu: float, n: int, state: InitState, *, nint: int = 250) -> float:
    """State wrapper for interaction_mod lattice_sum (backward-compatible API)."""
    case = state.case
    return _lattice_sum_core(
        d_mu,
        n,
        network=case.network,
        R=case.R,
        Rapparent=state.Rapparent,
        density=state.density,
        lattice_const=case.lattice_const,
        levels=case.Levels,
        nint=nint,
    )


def surf_const_coef_quadrupole(
    alphad: np.ndarray,
    alphaq: np.ndarray,
    state: InitState,
    ienergy: int,
    *,
    nint: int = 250,
) -> SurfaceConstitutive:
    """surf_const_coef_quadrupole in optics_mod.f90."""
    case = state.case
    e1 = state.eps_vacuum
    e2 = state.eps_substrate[ienergy]
    d = abs(case.tr - case.MPpos)
    density = state.density * case.R**2
    L = effective_lattice_L(
        network=case.network,
        lattice_const=case.lattice_const,
        R=case.R,
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
    state: InitState,
    ienergy: int,
    *,
    alphaq: np.ndarray | None = None,
    nint: int = 250,
) -> SurfaceConstitutive:
    """Island interaction surface coefficients (none / dipole / quadrupole)."""
    case = state.case
    interaction = case.interaction.strip().lower()
    if interaction == "none":
        return _surf_const_nointeract(alpha, state, ienergy)
    if interaction == "quadrupole":
        if alphaq is None:
            raise ValueError("alphaq required when interaction=quadrupole")
        return surf_const_coef_quadrupole(alpha, alphaq, state, ienergy, nint=nint)

    e1 = state.eps_vacuum
    d = abs(case.tr - case.MPpos)
    density = state.density * case.R**2
    L = effective_lattice_L(
        network=case.network,
        lattice_const=case.lattice_const,
        R=case.R,
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
            R=case.R,
            Rapparent=state.Rapparent,
            levels=case.Levels,
            density=state.density,
            above=state.above,
            nint=nint,
        )
    eta = (e1 - state.eps_substrate[ienergy]) / (e1 + state.eps_substrate[ienergy])
    sqr = np.sqrt(4.0 * PI / 5.0)

    if state.above:
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

    e2 = state.eps_substrate[ienergy]
    eps = (e2 - e1) / (e2 + e1)
    paral = alpha[0, 0] / (4 * PI * e2)
    factor = s_mp + eps * s_imp
    paral = 4 * PI * e2 * paral / (1.0 + paral * sqr / L**3 * factor)

    perp = alpha[0, 1] / (4 * PI * e2)
    factor = s_mp - eps * s_imp
    perp = 4 * PI * e2 * perp / (1.0 - 2.0 * perp * sqr / L**3 * factor)

    return SurfaceConstitutive(
        beta=density * perp / e2**2,
        gamma=density * paral,
        delta=-density * d * (paral + perp) / e2,
        tau=-density * d * paral * e1 / e2,
    )


def _surf_const_nointeract(alpha: np.ndarray, state: InitState, ienergy: int) -> SurfaceConstitutive:
    e1 = state.eps_vacuum
    d = abs(state.case.tr - state.case.MPpos)
    density = state.density * state.case.R**2
    if state.above:
        return SurfaceConstitutive(
            gamma=density * alpha[0, 0],
            beta=density * alpha[0, 1] / e1**2,
            delta=-density / e1 * np.sum(alpha[1, :] - d * alpha[0, :]),
            tau=-density * (alpha[1, 0] - d * alpha[0, 0]),
        )
    e2 = state.eps_substrate[ienergy]
    return SurfaceConstitutive(
        gamma=density * alpha[0, 0],
        beta=density * alpha[0, 1] / e2**2,
        delta=-density / e2 * np.sum(alpha[1, :] + d * alpha[0, :]),
        tau=-density * (e1 / e2) * (alpha[1, 0] + d * alpha[0, 0]),
    )
