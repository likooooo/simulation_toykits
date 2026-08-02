"""Step 2: oblate/prolate linear systems (oblate_mod / prolate_mod)."""

from __future__ import annotations

import numpy as np

from granfilm.common.constants import PI
from granfilm.common.linsolver import linsolve_granfilm
from granfilm.oblate_prolate.step0_init import SpheroidInitState
from granfilm.oblate_prolate.step1_integrals import SpheroidIntegrals
from granfilm.oblate_prolate.xz import XZField


def _kron(l: int, l1: int) -> float:
    return 1.0 if l == l1 else 0.0


def _build_system_above(
    m: int,
    ienergy: int,
    state: SpheroidInitState,
    integrals: SpheroidIntegrals,
    xz: XZField,
    zeta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mpo = state.case.Mpole_order
    tr = state.case.tr if state.above else abs(state.case.tr)
    xi0 = state.xi0
    inv_xi0 = 1.0 / xi0
    e1 = state.eps_vacuum
    e2 = state.eps_substrate[ienergy]
    e3 = state.eps_island[ienergy]
    e4 = e2
    sin0 = np.sin(state.theta0_calc)
    cos0 = np.cos(state.theta0_calc)
    exp0 = np.exp(-1j * state.phi0_calc)

    a = np.zeros((2 * mpo, 2 * mpo), dtype=np.complex128)
    b = np.zeros(2 * mpo, dtype=np.complex128)

    for l in range(1, mpo + 1):
        kronl = 1.0 if l == 1 else 0.0
        if m == 0:
            b[l - 1] = np.sqrt(4 * PI / 3) * cos0 * (
                e1 / e2 * inv_xi0 * xz.X[1, 0] * kronl
                + ((e1 - e2) / e2)
                * (
                    np.sqrt(3.0) * tr * zeta[l, 0, 0] * integrals.Q[l, 0, 0]
                    - inv_xi0 * xz.X[1, 0] * zeta[l, 1, 0] * integrals.Q[l, 1, 0]
                )
            )
            b[mpo + l - 1] = np.sqrt(4 * PI / 3) * cos0 * e1 * xz.dXdxi[1, 0] * kronl
        else:
            b[l - 1] = -np.sqrt(2 * PI / 3) * sin0 * exp0 * inv_xi0 * xz.X[1, 1] * kronl
            b[mpo + l - 1] = (
                -np.sqrt(2 * PI / 3)
                * sin0
                * exp0
                * xz.dXdxi[1, 1]
                * (e2 * kronl + (e1 - e2) * zeta[l, 1, 1] * integrals.Q[l, 1, 1])
            )

        for l1 in range(1, mpo + 1):
            kronl1 = _kron(l, l1)
            eps = np.array([(e1 - e2), 2 * e1, 2 * e1 * e2, e1 * (e1 - e2)], dtype=np.complex128) / (e1 + e2)
            z = zeta[l, l1, m]
            a[l - 1, l1 - 1] = eps[1] * xi0 ** (l1 + 1) * xz.Z[l1, m] * kronl1 - eps[0] * z * xi0 ** (l1 + 1) * (
                xz.Z[l1, m] * integrals.Q[l, l1, m] - (-1) ** (l1 + m) * integrals.V[l, l1, m]
            )
            a[mpo + l - 1, l1 - 1] = eps[2] * xi0 ** (l1 + 2) * xz.dZdxi[l1, m] * kronl1 + eps[3] * z * xi0 ** (
                l1 + 2
            ) * (xz.dZdxi[l1, m] * integrals.Q[l, l1, m] + (-1) ** (l1 + m) * integrals.dVdx[l, l1, m])

            eps_m = np.array([(e3 - e4), 2 * e3, 2 * e3 * e4, e3 * (e3 - e4)], dtype=np.complex128) / (e3 + e4)
            a[l - 1, mpo + l1 - 1] = -eps_m[1] * inv_xi0**l1 * xz.X[l1, m] * kronl1 + eps_m[0] * z * inv_xi0**l1 * (
                xz.X[l1, m] * integrals.Q[l, l1, m] - (-1) ** (l1 + m) * integrals.W[l, l1, m]
            )
            a[mpo + l - 1, mpo + l1 - 1] = (
                -eps_m[2] * inv_xi0 ** (l1 - 1) * xz.dXdxi[l1, m] * kronl1
                - eps_m[3]
                * z
                * inv_xi0 ** (l1 - 1)
                * (xz.dXdxi[l1, m] * integrals.Q[l, l1, m] + (-1) ** (l1 + m) * integrals.dWdx[l, l1, m])
            )
    return a, b


def _build_system_below(
    m: int,
    ienergy: int,
    state: SpheroidInitState,
    integrals: SpheroidIntegrals,
    xz: XZField,
    zeta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mpo = state.case.Mpole_order
    tr = abs(state.case.tr)
    xi0 = state.xi0
    inv_xi0 = 1.0 / xi0
    e1 = state.eps_vacuum
    e2 = state.eps_substrate[ienergy]
    e3 = state.eps_island[ienergy]
    e4 = e2
    sin0 = np.sin(state.theta0_calc)
    cos0 = np.cos(state.theta0_calc)
    exp0 = np.exp(-1j * state.phi0_calc)

    a = np.zeros((2 * mpo, 2 * mpo), dtype=np.complex128)
    b = np.zeros(2 * mpo, dtype=np.complex128)

    for l in range(1, mpo + 1):
        kronl = 1.0 if l == 1 else 0.0
        factor_l = (-1) ** (l + 1)
        if m == 0:
            b[l - 1] = np.sqrt(4 * PI / 3) * cos0 * (
                inv_xi0 * xz.X[1, 0] * kronl
                + ((e2 - e1) / e2)
                * factor_l
                * (
                    np.sqrt(3.0) * tr * zeta[l, 0, 0] * integrals.Q[l, 0, 0]
                    - inv_xi0 * xz.X[1, 0] * zeta[l, 1, 0] * integrals.Q[l, 1, 0]
                )
            )
            b[mpo + l - 1] = np.sqrt(4 * PI / 3) * cos0 * e1 * xz.dXdxi[1, 0] * kronl
        else:
            b[l - 1] = -np.sqrt(2 * PI / 3) * sin0 * exp0 * inv_xi0 * xz.X[1, 1] * kronl
            b[mpo + l - 1] = (
                -np.sqrt(2 * PI / 3)
                * sin0
                * exp0
                * xz.dXdxi[1, 1]
                * (e1 * kronl + (e2 - e1) * factor_l * zeta[l, 1, 1] * integrals.Q[l, 1, 1])
            )

        for l1 in range(1, mpo + 1):
            kronl1 = _kron(l, l1)
            factor = (-1) ** (l + l1)
            eps = np.array([(e2 - e1), 2 * e2, 2 * e2 * e1, e2 * (e2 - e1)], dtype=np.complex128) / (e2 + e1)
            z = zeta[l, l1, m]
            a[l - 1, l1 - 1] = eps[1] * xi0 ** (l1 + 1) * xz.Z[l1, m] * kronl1 - eps[0] * z * xi0 ** (l1 + 1) * factor * (
                xz.Z[l1, m] * integrals.Q[l, l1, m] - (-1) ** (l1 + m) * integrals.V[l, l1, m]
            )
            a[mpo + l - 1, l1 - 1] = eps[2] * xi0 ** (l1 + 2) * xz.dZdxi[l1, m] * kronl1 + eps[3] * z * xi0 ** (
                l1 + 2
            ) * factor * (
                xz.dZdxi[l1, m] * integrals.Q[l, l1, m] + (-1) ** (l1 + m) * integrals.dVdx[l, l1, m]
            )

            eps_m = np.array([(e4 - e3), 2 * e4, 2 * e4 * e3, e4 * (e4 - e3)], dtype=np.complex128) / (e4 + e3)
            a[l - 1, mpo + l1 - 1] = -eps_m[1] * inv_xi0**l1 * xz.X[l1, m] * kronl1 + eps_m[0] * z * inv_xi0**l1 * factor * (
                xz.X[l1, m] * integrals.Q[l, l1, m] - (-1) ** (l1 + m) * integrals.W[l, l1, m]
            )
            a[mpo + l - 1, mpo + l1 - 1] = (
                -eps_m[2] * inv_xi0 ** (l1 - 1) * xz.dXdxi[l1, m] * kronl1
                - eps_m[3]
                * z
                * inv_xi0 ** (l1 - 1)
                * factor
                * (xz.dXdxi[l1, m] * integrals.Q[l, l1, m] + (-1) ** (l1 + m) * integrals.dWdx[l, l1, m])
            )
    return a, b


def step2_solve_multipoles(
    state: SpheroidInitState,
    integrals: SpheroidIntegrals,
    xz: XZField,
    zeta: np.ndarray,
    ienergy: int,
) -> np.ndarray:
    """Return mpoled[2,2]: dipole (l=1) and quadrupole (l=2) coefficients for m=0,1."""
    build = _build_system_above if state.above else _build_system_below
    a0, b0 = build(0, ienergy, state, integrals, xz, zeta)
    a1, b1 = build(1, ienergy, state, integrals, xz, zeta)
    bd0 = linsolve_granfilm(a0, b0, state.case.epslin)
    bd1 = linsolve_granfilm(a1, b1, state.case.epslin)
    return np.array([[bd0[0], bd1[0]], [bd0[1], bd1[1]]], dtype=np.complex128)
