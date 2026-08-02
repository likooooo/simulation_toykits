"""Step 3: multipole coefficients to polarizabilities (Spheroid optics_mod)."""

from __future__ import annotations

import numpy as np

from granfilm.common.constants import PI
from granfilm.oblate_prolate.step0_init import SpheroidInitState


def step3_polarizabilities(mpoled: np.ndarray, state: SpheroidInitState, ienergy: int) -> np.ndarray:
    """Return alpha[2,2]: (order 1/2) x (parallel/perp), dimensionless."""
    e1 = state.eps_vacuum
    c0 = np.cos(state.theta0_calc)
    s0 = np.sin(state.theta0_calc)
    exp0 = np.exp(-1j * state.phi0_calc)

    a10, a11 = mpoled[0, 0], mpoled[0, 1]
    a20, a21 = mpoled[1, 0], mpoled[1, 1]

    alpha = np.zeros((2, 2), dtype=np.complex128)
    if state.above:
        alpha[0, 0] = -4 * PI * e1 * a11 / (np.sqrt(2 * PI / 3) * s0 * exp0)
        alpha[0, 1] = 2 * PI * e1 * a10 / (np.sqrt(PI / 3) * c0)
        alpha[1, 0] = -4 * PI * e1 * a21 / (np.sqrt(6 * PI / 5) * s0 * exp0)
        alpha[1, 1] = PI * e1 * a20 / (np.sqrt(PI / 5) * c0)
    else:
        e2 = state.eps_substrate[ienergy]
        alpha[0, 0] = -4 * PI * e2 * a11 / (np.sqrt(2 * PI / 3) * s0 * exp0)
        alpha[0, 1] = 2 * PI * e2 * a10 / ((e1 / e2) * np.sqrt(PI / 3) * c0)
        alpha[1, 0] = -4 * PI * e2 * a21 / (np.sqrt(6 * PI / 5) * s0 * exp0)
        alpha[1, 1] = PI * e2 * a20 / ((e1 / e2) * np.sqrt(PI / 5) * c0)
    return alpha
