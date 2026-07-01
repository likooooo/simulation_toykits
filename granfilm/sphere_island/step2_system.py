"""Step 2: multipole linear system (matrix_system_mod.f90)."""

from __future__ import annotations

import numpy as np

from granfilm.common.constants import PI
from granfilm.common.linsolver import linsolve_granfilm
from granfilm.sphere_island.step0_init import InitState
from granfilm.sphere_island.step1_integrals import IntegralsVolume


def _matrix_system_above(
    m: int,
    ienergy: int,
    state: InitState,
    Int: IntegralsVolume,
    int_tr1: IntegralsVolume,
    zeta: np.ndarray,
) -> np.ndarray:
    mpo = state.case.Mpole_order
    e1 = state.eps_vacuum
    e2 = state.eps_substrate[ienergy]
    e3 = state.eps_island[ienergy]
    e4 = e2
    A = np.zeros((2 * mpo, 2 * mpo), dtype=np.complex128)

    for l1 in range(1, mpo + 1):
        for l2 in range(1, mpo + 1):
            eps = np.array(
                [
                    (e1 - e2),
                    2 * e1,
                    2 * e1 * e2,
                    e1 * (e1 - e2),
                ],
                dtype=np.complex128,
            ) / (e1 + e2)
            z = zeta[l1, l2, m]
            A[l1 - 1, l2 - 1] = z * (
                Int.K[l1, l2, m, 0]
                + ((-1) ** (l2 + m)) * eps[0] * Int.K[l1, l2, m, 1]
                + eps[1] * (int_tr1.K[l1, l2, m, 0] - Int.K[l1, l2, m, 0])
            )
            A[mpo + l1 - 1, l2 - 1] = z * (
                eps[2] * int_tr1.L[l1, l2, m, 0]
                + eps[3]
                * (Int.L[l1, l2, m, 0] + ((-1) ** (l2 + m)) * Int.L[l1, l2, m, 1])
            )

            eps_m = np.array(
                [
                    (e3 - e4),
                    2 * e3,
                    2 * e3 * e4,
                    e3 * (e3 - e4),
                ],
                dtype=np.complex128,
            ) / (e3 + e4)
            A[l1 - 1, mpo + l2 - 1] = -z * (
                Int.M[l1, l2, m, 0]
                + ((-1) ** (l2 + m)) * eps_m[0] * Int.M[l1, l2, m, 1]
                + eps_m[1] * (int_tr1.M[l1, l2, m, 0] - Int.M[l1, l2, m, 0])
            )
            A[mpo + l1 - 1, mpo + l2 - 1] = -z * (
                eps_m[2] * int_tr1.N[l1, l2, m, 0]
                + eps_m[3]
                * (Int.N[l1, l2, m, 0] + ((-1) ** (l2 + m)) * Int.N[l1, l2, m, 1])
            )
    return A


def _right_dipole_above(
    m: int,
    ienergy: int,
    state: InitState,
    Int: IntegralsVolume,
    zeta: np.ndarray,
) -> np.ndarray:
    mpo = state.case.Mpole_order
    tr = state.case.tr
    sin0 = np.sin(state.theta0_calc)
    cos0 = np.cos(state.theta0_calc)
    exp0 = np.exp(-1j * state.phi0_calc)
    e1 = state.eps_vacuum
    e2 = state.eps_substrate[ienergy]
    b = np.zeros(2 * mpo, dtype=np.complex128)

    for l1 in range(1, mpo + 1):
        kronl = 1.0 if l1 == 1 else 0.0
        if m == 0:
            b[l1 - 1] = np.sqrt(4.0 * PI / 3.0) * cos0 * (
                e1 / e2 * kronl
                + (e1 - e2)
                / e2
                * (
                    np.sqrt(3.0) * tr * zeta[l1, 0, 0] * Int.Q[l1, 0, 0]
                    - zeta[l1, 1, 0] * Int.Q[l1, 1, 0]
                )
            )
            b[mpo + l1 - 1] = np.sqrt(4.0 * PI / 3.0) * e1 * cos0 * kronl
        elif m == 1:
            b[l1 - 1] = -np.sqrt(2.0 * PI / 3.0) * sin0 * kronl * exp0
            b[mpo + l1 - 1] = -np.sqrt(2.0 * PI / 3.0) * sin0 * exp0 * (
                e2 * kronl + (e1 - e2) * zeta[l1, 1, 1] * Int.Q[l1, 1, 1]
            )
    return b


def _right_quadrupole_above(
    m: int,
    ienergy: int,
    state: InitState,
    Int: IntegralsVolume,
    zeta: np.ndarray,
) -> np.ndarray:
    """Right_Quadrupole_above in matrix_system_mod.f90 (formulas 10-71 … 10-77)."""
    mpo = state.case.Mpole_order
    tr = state.case.tr
    e1 = state.eps_vacuum
    e2 = state.eps_substrate[ienergy]
    sqr = np.sqrt(2.0 * PI / 15.0)
    b = np.zeros(2 * mpo, dtype=np.complex128)

    for l1 in range(1, mpo + 1):
        kronl1 = 1.0 if l1 == 1 else 0.0
        kronl2 = 1.0 if l1 == 2 else 0.0
        if m == 0:
            b[l1 - 1] = -2.0 * np.sqrt(4.0 * PI / 3.0) * (
                np.sqrt(3.0 / 5.0) / 2.0 * kronl2
                + (e1 / e2 - 1.0)
                * tr
                * (
                    np.sqrt(3.0) * tr * zeta[l1, 0, 0] * Int.Q[l1, 0, 0]
                    + kronl1
                    - zeta[l1, 1, 0] * Int.Q[l1, 1, 0]
                )
            )
            b[mpo + l1 - 1] = 4.0 * np.sqrt(PI / 5.0) * (
                (e2 - e1) * zeta[l1, 2, 0] * Int.Q[l1, 2, 0]
                - e2 * kronl2
                + np.sqrt(5.0 / 3.0) * (e2 - e1) * tr * (kronl1 - zeta[l1, 1, 0] * Int.Q[l1, 1, 0])
            )
        elif m == 1:
            b[l1 - 1] = -sqr * (1.0 - 1j) * (
                e1 / e2 * kronl2
                - (e1 / e2 - 1.0)
                * (
                    zeta[l1, 2, 1] * Int.Q[l1, 2, 1]
                    + np.sqrt(5.0) * tr * (kronl1 - zeta[l1, 1, 1] * Int.Q[l1, 1, 1])
                )
            )
            b[mpo + l1 - 1] = -sqr * (1.0 - 1j) * (
                2.0 * e1 * kronl2
                + np.sqrt(5.0) * (e2 - e1) * tr * (kronl1 - zeta[l1, 1, 1] * Int.Q[l1, 1, 1])
            )
        elif m == 2:
            if l1 == 1:
                continue
            b[l1 - 1] = -sqr * 1j * kronl2
            b[mpo + l1 - 1] = -2.0 * sqr * 1j * (
                e2 * kronl2 - (e2 - e1) * zeta[l1, 2, 2] * Int.Q[l1, 2, 2]
            )
    return b


def _matrix_system_below(
    m: int,
    ienergy: int,
    state: InitState,
    Int: IntegralsVolume,
    int_tr1: IntegralsVolume,
    zeta: np.ndarray,
) -> np.ndarray:
    """Matrix_system_below in matrix_system_mod.f90 (e1↔e2, factor=(-1)^(l1+l2))."""
    mpo = state.case.Mpole_order
    e1 = state.eps_vacuum
    e2 = state.eps_substrate[ienergy]
    e3 = state.eps_island[ienergy]
    e4 = e2
    A = np.zeros((2 * mpo, 2 * mpo), dtype=np.complex128)

    for l1 in range(1, mpo + 1):
        for l2 in range(1, mpo + 1):
            factor = (-1) ** (l1 + l2)
            eps = np.array(
                [
                    (e2 - e1),
                    2 * e2,
                    2 * e2 * e1,
                    e2 * (e2 - e1),
                ],
                dtype=np.complex128,
            ) / (e2 + e1)
            z = zeta[l1, l2, m]
            inner_c = (
                Int.K[l1, l2, m, 0]
                + ((-1) ** (l2 + m)) * eps[0] * Int.K[l1, l2, m, 1]
                + eps[1] * (int_tr1.K[l1, l2, m, 0] - Int.K[l1, l2, m, 0])
            )
            A[l1 - 1, l2 - 1] = factor * z * inner_c
            A[mpo + l1 - 1, l2 - 1] = factor * z * (
                eps[2] * int_tr1.L[l1, l2, m, 0]
                + eps[3]
                * (Int.L[l1, l2, m, 0] + ((-1) ** (l2 + m)) * Int.L[l1, l2, m, 1])
            )

            eps_m = np.array(
                [
                    (e4 - e3),
                    2 * e4,
                    2 * e4 * e3,
                    e4 * (e4 - e3),
                ],
                dtype=np.complex128,
            ) / (e4 + e3)
            inner_d = (
                Int.M[l1, l2, m, 0]
                + ((-1) ** (l2 + m)) * eps_m[0] * Int.M[l1, l2, m, 1]
                + eps_m[1] * (int_tr1.M[l1, l2, m, 0] - Int.M[l1, l2, m, 0])
            )
            A[l1 - 1, mpo + l2 - 1] = -factor * z * inner_d
            A[mpo + l1 - 1, mpo + l2 - 1] = -factor * z * (
                eps_m[2] * int_tr1.N[l1, l2, m, 0]
                + eps_m[3]
                * (Int.N[l1, l2, m, 0] + ((-1) ** (l2 + m)) * Int.N[l1, l2, m, 1])
            )
    return A


def _right_dipole_below(
    m: int,
    ienergy: int,
    state: InitState,
    Int: IntegralsVolume,
    zeta: np.ndarray,
) -> np.ndarray:
    """Right_Dipole_below in matrix_system_mod.f90."""
    mpo = state.case.Mpole_order
    tr = state.case.tr
    sin0 = np.sin(state.theta0_calc)
    cos0 = np.cos(state.theta0_calc)
    exp0 = np.exp(-1j * state.phi0_calc)
    e1 = state.eps_vacuum
    e2 = state.eps_substrate[ienergy]
    b = np.zeros(2 * mpo, dtype=np.complex128)

    for l1 in range(1, mpo + 1):
        kronl = 1.0 if l1 == 1 else 0.0
        factor = (-1) ** (l1 + 1)
        if m == 0:
            b[l1 - 1] = np.sqrt(4.0 * PI / 3.0) * cos0 * (
                kronl
                + (e2 - e1)
                / e2
                * factor
                * (
                    -np.sqrt(3.0) * tr * zeta[l1, 0, 0] * Int.Q[l1, 0, 0]
                    - zeta[l1, 1, 0] * Int.Q[l1, 1, 0]
                )
            )
            b[mpo + l1 - 1] = np.sqrt(4.0 * PI / 3.0) * e1 * cos0 * kronl
        elif m == 1:
            b[l1 - 1] = -np.sqrt(2.0 * PI / 3.0) * sin0 * kronl * exp0
            b[mpo + l1 - 1] = -np.sqrt(2.0 * PI / 3.0) * sin0 * exp0 * (
                e1 * kronl + (e2 - e1) * factor * zeta[l1, 1, 1] * Int.Q[l1, 1, 1]
            )
    return b


def _right_quadrupole_below(
    m: int,
    ienergy: int,
    state: InitState,
    Int: IntegralsVolume,
    zeta: np.ndarray,
) -> np.ndarray:
    """Right_Quadrupole_below in matrix_system_mod.f90 (tr = -|tr|)."""
    mpo = state.case.Mpole_order
    tr = -abs(state.case.tr)
    e1 = state.eps_vacuum
    e2 = state.eps_substrate[ienergy]
    sqr = np.sqrt(2.0 * PI / 15.0)
    b = np.zeros(2 * mpo, dtype=np.complex128)

    for l1 in range(1, mpo + 1):
        kronl1 = 1.0 if l1 == 1 else 0.0
        kronl2 = 1.0 if l1 == 2 else 0.0
        if m == 0:
            b[l1 - 1] = -2.0 * np.sqrt(4.0 * PI / 3.0) * (
                np.sqrt(3.0 / 5.0) / 2.0 * kronl2
                + (e2 / e1 - 1.0)
                * tr
                * (
                    np.sqrt(3.0) * tr * zeta[l1, 0, 0] * Int.Q[l1, 0, 0]
                    + kronl1
                    - zeta[l1, 1, 0] * Int.Q[l1, 1, 0]
                )
            )
            b[mpo + l1 - 1] = 4.0 * np.sqrt(PI / 5.0) * (
                (e1 - e2) * zeta[l1, 2, 0] * Int.Q[l1, 2, 0]
                - e1 * kronl2
                + np.sqrt(5.0 / 3.0) * (e1 - e2) * tr * (kronl1 - zeta[l1, 1, 0] * Int.Q[l1, 1, 0])
            )
        elif m == 1:
            b[l1 - 1] = -sqr * (1.0 - 1j) * (
                e2 / e1 * kronl2
                - (e2 / e1 - 1.0)
                * (
                    zeta[l1, 2, 1] * Int.Q[l1, 2, 1]
                    + np.sqrt(5.0) * tr * (kronl1 - zeta[l1, 1, 1] * Int.Q[l1, 1, 1])
                )
            )
            b[mpo + l1 - 1] = -sqr * (1.0 - 1j) * (
                2.0 * e2 * kronl2
                + np.sqrt(5.0) * (e1 - e2) * tr * (kronl1 - zeta[l1, 1, 1] * Int.Q[l1, 1, 1])
            )
        elif m == 2:
            if l1 == 1:
                continue
            b[l1 - 1] = -sqr * 1j * kronl2
            b[mpo + l1 - 1] = -2.0 * sqr * 1j * (
                e1 * kronl2 - (e1 - e2) * zeta[l1, 2, 2] * Int.Q[l1, 2, 2]
            )
    return b


def _matrix_system(
    m: int,
    ienergy: int,
    state: InitState,
    Int: IntegralsVolume,
    int_tr1: IntegralsVolume,
    zeta: np.ndarray,
) -> np.ndarray:
    if state.above:
        return _matrix_system_above(m, ienergy, state, Int, int_tr1, zeta)
    return _matrix_system_below(m, ienergy, state, Int, int_tr1, zeta)


def _right_dipole(
    m: int,
    ienergy: int,
    state: InitState,
    Int: IntegralsVolume,
    zeta: np.ndarray,
) -> np.ndarray:
    if state.above:
        return _right_dipole_above(m, ienergy, state, Int, zeta)
    return _right_dipole_below(m, ienergy, state, Int, zeta)


def _right_quadrupole(
    m: int,
    ienergy: int,
    state: InitState,
    Int: IntegralsVolume,
    zeta: np.ndarray,
) -> np.ndarray:
    if state.above:
        return _right_quadrupole_above(m, ienergy, state, Int, zeta)
    return _right_quadrupole_below(m, ienergy, state, Int, zeta)


def step2_solve_multipoles(
    state: InitState,
    Int: IntegralsVolume,
    int_tr1: IntegralsVolume,
    zeta: np.ndarray,
    ienergy: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Solve dipole and (when interaction=quadrupole) quadrupole multipole coefficients.
    Returns (mpoled, mpoleq) where mpoled shape (2, 2) matches bd(1:2, 0:1).
    mpoleq is shape (3,) or None when interaction is not quadrupole.
    """
    mpo = state.case.Mpole_order
    interaction = state.case.interaction.strip().lower()
    mpoled = np.zeros((2, 2), dtype=np.complex128)
    mpoleq: np.ndarray | None = None

    for m in (0, 1):
        A = _matrix_system(m, ienergy, state, Int, int_tr1, zeta)
        b = _right_dipole(m, ienergy, state, Int, zeta)
        x = linsolve_granfilm(A, b, epslin=state.case.epslin)
        mpoled[0, m] = x[0]
        if mpo >= 2:
            mpoled[1, m] = x[1]

    if interaction == "quadrupole":
        if mpo < 2:
            raise ValueError("quadrupole interaction requires Mpole_order >= 2")
        mpoleq = np.zeros(3, dtype=np.complex128)
        for m in (0, 1):
            A = _matrix_system(m, ienergy, state, Int, int_tr1, zeta)
            bq = _right_quadrupole(m, ienergy, state, Int, zeta)
            xq = linsolve_granfilm(A, bq, epslin=state.case.epslin)
            mpoleq[m] = xq[1]

        A = _matrix_system(2, ienergy, state, Int, int_tr1, zeta)
        bq = _right_quadrupole(2, ienergy, state, Int, zeta)
        n_red = 2 * mpo - 2
        Atmp = np.zeros((n_red, n_red), dtype=np.complex128)
        btmp = np.zeros(n_red, dtype=np.complex128)
        Atmp[: mpo - 1, : mpo - 1] = A[1:mpo, 1:mpo]
        Atmp[: mpo - 1, mpo - 1 :] = A[1:mpo, mpo + 1 : 2 * mpo]
        Atmp[mpo - 1 :, : mpo - 1] = A[mpo + 1 : 2 * mpo, 1:mpo]
        Atmp[mpo - 1 :, mpo - 1 :] = A[mpo + 1 : 2 * mpo, mpo + 1 : 2 * mpo]
        btmp[: mpo - 1] = bq[1:mpo]
        btmp[mpo - 1 :] = bq[mpo + 1 : 2 * mpo]
        xtmp = linsolve_granfilm(Atmp, btmp, epslin=state.case.epslin)
        mpoleq[2] = xtmp[0]

    return mpoled, mpoleq


def step2_system_matrix(
    state: InitState,
    Int: IntegralsVolume,
    int_tr1: IntegralsVolume,
    zeta: np.ndarray,
    ienergy: int,
    m: int = 0,
) -> np.ndarray:
    """Expose A matrix for visualization (default m=0)."""
    return _matrix_system(m, ienergy, state, Int, int_tr1, zeta)
