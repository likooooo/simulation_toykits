"""Step 3: multipole coefficients to polarizabilities (optics_mod polarizabilities)."""

from __future__ import annotations

import numpy as np

from granfilm.common.constants import PI
from granfilm.sphere_island.step0_init import InitState


def step3_polarizabilities(mpoled: np.ndarray, state: InitState, ienergy: int) -> np.ndarray:
    """
    Return alpha[2,2]: (order 1/2) x (parallel/perp).
    Matches optics_mod.f90 polarizabilities (Eq. 8.2.29 above, 8.3.7 below).
    """
    e1 = state.eps_vacuum
    c0 = np.cos(state.theta0_calc)
    s0 = np.sin(state.theta0_calc)
    exp0 = np.exp(-1j * state.phi0_calc)

    A10 = mpoled[0, 0]
    A11 = mpoled[0, 1]
    A20 = mpoled[1, 0]
    A21 = mpoled[1, 1]

    alpha = np.zeros((2, 2), dtype=np.complex128)
    if state.above:
        alpha[0, 0] = -4 * PI * e1 * A11 / (np.sqrt(2 * PI / 3) * s0 * exp0)
        alpha[0, 1] = 2 * PI * e1 * A10 / (np.sqrt(PI / 3) * c0)
        alpha[1, 0] = -4 * PI * e1 * A21 / (np.sqrt(6 * PI / 5) * s0 * exp0)
        alpha[1, 1] = PI * e1 * A20 / (np.sqrt(PI / 5) * c0)
    else:
        e2 = state.eps_substrate[ienergy]
        alpha[0, 0] = -4 * PI * e2 * A11 / (np.sqrt(2 * PI / 3) * s0 * exp0)
        alpha[0, 1] = 2 * PI * e2 * A10 / ((e1 / e2) * np.sqrt(PI / 3) * c0)
        alpha[1, 0] = -4 * PI * e2 * A21 / (np.sqrt(6 * PI / 5) * s0 * exp0)
        alpha[1, 1] = PI * e2 * A20 / ((e1 / e2) * np.sqrt(PI / 5) * c0)
    return alpha


def step3_polarizabilities_quadrupole(
    mpoleq: np.ndarray,
    state: InitState,
    ienergy: int,
) -> np.ndarray:
    """polarizabilities_quadrupole in optics_mod.f90 (Eq. 10.88)."""
    e1 = state.eps_vacuum
    sqr = np.sqrt(6.0 * PI / 5.0)
    a20, a21, a22 = mpoleq[0], mpoleq[1], mpoleq[2]
    alphaq = np.zeros(3, dtype=np.complex128)
    if state.above:
        alphaq[0] = -PI * e1 * a20 / (sqr * np.sqrt(2.0 / 3.0)) + 3 * PI * e1 * a22 / (1j * sqr)
        alphaq[1] = -4 * PI * e1 * a21 / sqr / (1.0 - 1j) + 4 * PI * e1 * a22 / (1j * sqr)
        alphaq[2] = -2 * PI * e1 * a22 / (1j * sqr)
    else:
        e2 = state.eps_substrate[ienergy]
        alphaq[0] = -PI * e2 * a20 / (sqr * np.sqrt(2.0 / 3.0)) - 3 * PI * e2 * a22 / (1j * sqr)
        alphaq[1] = -4 * PI * e2 * a21 / sqr / (1.0 + 1j) - 4 * PI * e2 * a22 / (1j * sqr)
        alphaq[2] = 2 * PI * e2 * a22 / (1j * sqr)
    return alphaq
