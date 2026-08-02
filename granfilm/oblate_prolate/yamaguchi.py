"""Yamaguchi spheroid polarizabilities (GranFilm yamaguchi.f90)."""

from __future__ import annotations

import numpy as np

from granfilm.common.constants import PI
from granfilm.common.legendre import gauss_legendre
from granfilm.oblate_prolate.step0_init import SpheroidInitState

_QUADPACK_INFINITE_UPPER = 100.0


def _quadpack_infinite(f, bound: float, *, nint: int = 250) -> float:
    """Match GranFilm quadpack_stub dqagi(inf=1): integrate from bound to bound+100."""
    x, w = gauss_legendre(bound, bound + _QUADPACK_INFINITE_UPPER, nint)
    return float(np.sum(w * f(x)))


def _fk_integrand(q: np.ndarray, radius: np.ndarray, x: float) -> np.ndarray:
    a, b, c = radius
    return 1.0 / ((q + x**2) * np.sqrt((a**2 + q) * (b**2 + q) * (c**2 + q)))


def _depolarization_integrals_oblate(R: np.ndarray, *, nint: int = 250) -> np.ndarray:
    """Return L[2,2] depolarization integrals for oblate coated shells."""
    L = np.zeros((2, 2), dtype=np.float64)
    for j in range(2):
        radius = np.array([R[0, j], R[0, j], R[1, j]], dtype=np.float64)
        for i in range(2):
            x = R[i, j]

            def f(q: np.ndarray) -> np.ndarray:
                return _fk_integrand(q, radius, x)

            L[i, j] = _quadpack_infinite(f, 0.0, nint=nint)
    L[0, :] *= R[0, :] ** 2 * R[1, :] / 2.0
    L[1, :] *= R[0, :] ** 2 * R[1, :] / 2.0
    return L


def _depolarization_integrals_prolate(R: np.ndarray, *, nint: int = 250) -> np.ndarray:
    """Return L[2,2] depolarization integrals for prolate coated shells."""
    L = np.zeros((2, 2), dtype=np.float64)
    for j in range(2):
        radius = np.array([R[0, j], R[1, j], R[1, j]], dtype=np.float64)
        for i in range(2):
            x = R[i, j]

            def f(q: np.ndarray) -> np.ndarray:
                return _fk_integrand(q, radius, x)

            L[i, j] = _quadpack_infinite(f, 0.0, nint=nint)
    L[0, :] *= R[1, :] ** 2 * R[0, :] / 2.0
    L[1, :] *= R[1, :] ** 2 * R[0, :] / 2.0
    return L


def _oblate_yamaguchi_geometry(xi0: float, a_norm: float) -> tuple[float, float, float, float, float, float]:
    arctan = np.arctan(1.0 / xi0)
    volume = 4.0 * PI * a_norm**3 * xi0 * (1.0 + xi0**2) / 3.0
    laz = (1.0 + xi0**2) * (1.0 - xi0 * arctan)
    lap = (1.0 + xi0**2) * (xi0 * arctan - xi0**2 / (1.0 + xi0**2)) / 2.0
    laz1 = 3.0 * (1.0 + xi0**2) * (xi0 * (1.0 + 3.0 * xi0**2) * arctan - 3.0 * xi0**2) / 2.0
    lap1 = (1.0 + 2.0 * xi0**2) * (2.0 + 3.0 * xi0**2 - 3.0 * (1.0 + xi0**2) * xi0 * arctan) / 2.0
    la11 = (3.0 * (1.0 + xi0**2) ** 2 * xi0 * arctan - xi0**2 * (5.0 + 3.0 * xi0**2)) / 4.0
    return volume, laz, lap, laz1, lap1, la11


def _prolate_yamaguchi_geometry(xi0: float, a_norm: float) -> tuple[float, float, float, float, float, float]:
    loga = np.log((xi0 + 1.0) / (xi0 - 1.0)) / 2.0
    volume = 4.0 * PI * a_norm**3 * xi0 * (xi0**2 - 1.0) / 3.0
    laz = (1.0 - xi0**2) * (1.0 - xi0 * loga)
    lap = (1.0 - xi0**2) * (xi0**2 / (1.0 - xi0**2) + xi0 * loga) / 2.0
    laz1 = 3.0 * (1.0 - xi0**2) * ((1.0 - 3.0 * xi0**2) * xi0 * loga + 3.0 * xi0**2) / 2.0
    lap1 = (1.0 - 2.0 * xi0**2) * (-3.0 * (1.0 - xi0**2) * xi0 * loga + 2.0 - 3.0 * xi0**2) / 2.0
    la11 = (3.0 * (1.0 - xi0**2) ** 2 * xi0 * loga + xi0**2 * (5.0 - 3.0 * xi0**2)) / 4.0
    return volume, laz, lap, laz1, lap1, la11


def _quadrupole_tmp_oblate(
    e1: complex,
    e3: complex,
    a_norm: float,
    xi0: float,
    volume: float,
    laz1: float,
    lap1: float,
    la11: float,
) -> tuple[complex, complex, complex]:
    tmp3 = e1 * (e3 - e1) * (1.0 + 3.0 * xi0**2) * a_norm**2 * volume / 5.0 / (e1 + laz1 * (e3 - e1))
    tmp4 = e1 * (e3 - e1) * (1.0 + 2.0 * xi0**2) * a_norm**2 * volume / 5.0 / (e1 + lap1 * (e3 - e1))
    tmp5 = e1 * (e3 - e1) * (1.0 + xi0**2) * a_norm**2 * volume / 3.0 / (e1 + la11 * (e3 - e1))
    return tmp3, tmp4, tmp5


def _quadrupole_tmp_prolate(
    e1: complex,
    e3: complex,
    a_norm: float,
    xi0: float,
    volume: float,
    laz1: float,
    lap1: float,
    la11: float,
) -> tuple[complex, complex, complex]:
    tmp3 = e1 * (e3 - e1) * (3.0 * xi0**2 - 1.0) * a_norm**2 * volume / 5.0 / (e1 + laz1 * (e3 - e1))
    tmp4 = e1 * (e3 - e1) * (2.0 * xi0**2 - 1.0) * a_norm**2 * volume / 5.0 / (e1 + lap1 * (e3 - e1))
    tmp5 = e1 * (e3 - e1) * (xi0**2 - 1.0) * a_norm**2 * volume / 3.0 / (e1 + la11 * (e3 - e1))
    return tmp3, tmp4, tmp5


def _apply_mpole_order(
    mpole_order: int,
    tmp1: complex,
    tmp2: complex,
    tmp3: complex,
    tmp4: complex,
    imgs: complex,
    d: float,
) -> np.ndarray:
    alpha = np.zeros((2, 2), dtype=np.complex128)
    if mpole_order == 0:
        alpha[0, 0] = tmp1
        alpha[0, 1] = tmp2
    elif mpole_order == 1:
        alpha[0, 0] = tmp1 / (1.0 + imgs * tmp1)
        alpha[0, 1] = tmp2 / (1.0 + 2.0 * imgs * tmp2)
        alpha[1, 0] = -d * tmp1 / (1.0 + imgs * tmp1)
        alpha[1, 1] = -d * tmp2 / (1.0 + 2.0 * imgs * tmp2)
    elif mpole_order == 2:
        dz = 1.0 + 2.0 * imgs * tmp2 + (3.0 * imgs / (2.0 * d**2)) * (2.0 + imgs * tmp2) * tmp3
        dp = 1.0 + imgs * tmp1 + (3.0 * imgs / (4.0 * d**2)) * (4.0 + imgs * tmp1) * tmp4
        alpha[0, 0] = tmp1 * (1.0 + 3.0 * imgs / d**2 * tmp4) / dp
        alpha[0, 1] = -3.0 * imgs / d * tmp1 * tmp4 / 2.0 / dp
        alpha[1, 0] = tmp2 * (1.0 + 3.0 * imgs / d**2 * tmp3) / dz
        alpha[1, 1] = -3.0 * imgs / d * tmp2 * tmp3 / 2.0 / dz
    else:
        raise ValueError(f"unsupported Yamaguchi Mpole_order={mpole_order}")
    return alpha


def spheroid_yamaguchi(state: SpheroidInitState) -> np.ndarray:
    """Spheroid_Yamaguchi: polarizabilities for geometry Yamaguchi."""
    case = state.case
    xi0 = state.xi0
    a_norm = state.a / state.R
    e1 = state.eps_vacuum
    d = abs(case.tr)
    mpole_order = case.Mpole_order
    n = len(state.energy)
    alpha_all = np.zeros((n, 2, 2), dtype=np.complex128)

    if state.island_type == "oblate":
        volume, laz, lap, laz1, lap1, la11 = _oblate_yamaguchi_geometry(xi0, a_norm)
    elif state.island_type == "prolate":
        volume, laz, lap, laz1, lap1, la11 = _prolate_yamaguchi_geometry(xi0, a_norm)
    else:
        raise ValueError(f"unsupported island_type for Yamaguchi: {state.island_type}")

    for i in range(n):
        e3 = state.eps_island[i]
        e2 = state.eps_substrate[i]
        tmp1 = e1 * (e3 - e1) * volume / (e1 + lap * (e3 - e1))
        tmp2 = e1 * (e3 - e1) * volume / (e1 + laz * (e3 - e1))
        if state.island_type == "oblate":
            tmp3, tmp4, tmp5 = _quadrupole_tmp_oblate(e1, e3, a_norm, xi0, volume, laz1, lap1, la11)
        else:
            tmp3, tmp4, tmp5 = _quadrupole_tmp_prolate(e1, e3, a_norm, xi0, volume, laz1, lap1, la11)
        imgs = (e1 - e2) / (e1 + e2) / (32.0 * PI * e1 * d**3)
        alpha_all[i] = _apply_mpole_order(mpole_order, tmp1, tmp2, tmp3, tmp4, imgs, d)
    return alpha_all


def coating_yamaguchi(state: SpheroidInitState, *, nint: int = 250) -> np.ndarray:
    """Coating_Yamaguchi: coated spheroid polarizabilities at dipolar order."""
    if state.eps_coating is None:
        raise ValueError("coated geometry requires eps_coating in SpheroidInitState")

    case = state.case
    e1 = state.eps_vacuum
    d = abs(case.tr)
    t = case.thickness / state.R
    mpole_order = case.Mpole_order
    if mpole_order not in (0, 1):
        raise ValueError(f"Coating_Yamaguchi supports Mpole_order 0 or 1, got {mpole_order}")

    R = np.zeros((2, 2), dtype=np.float64)
    R[0, 1] = case.R_par / state.R
    R[0, 0] = R[0, 1] - t
    R[1, 1] = case.R_per / state.R
    R[1, 0] = R[1, 1] - t

    if state.island_type == "oblate":
        volume = 4.0 * PI * R[0, 1] ** 2 * R[1, 1] / 3.0
        f = R[0, 0] ** 2 * R[1, 0] / (R[0, 1] ** 2 * R[1, 1])
        L = _depolarization_integrals_oblate(R, nint=nint)
    elif state.island_type == "prolate":
        volume = 4.0 * PI * R[1, 1] ** 2 * R[0, 1] / 3.0
        f = R[1, 0] ** 2 * R[0, 0] / (R[1, 1] ** 2 * R[0, 1])
        L = _depolarization_integrals_prolate(R, nint=nint)
    else:
        raise ValueError(f"unsupported island_type for coated Yamaguchi: {state.island_type}")

    n = len(state.energy)
    alpha_all = np.zeros((n, 2, 2), dtype=np.complex128)
    for i in range(n):
        e2 = state.eps_substrate[i]
        e3 = state.eps_island[i]
        ec = state.eps_coating[i]
        imgs = (e1 - e2) / (e1 + e2) / (32.0 * PI * e1 * d**3)

        num_p = (ec - e1) * (ec + (e3 - ec) * (L[0, 0] - f * L[0, 1])) + f * ec * (e3 - ec)
        den_p = (ec + (e3 - ec) * (L[0, 0] - f * L[0, 1])) * (e1 + (ec - e1) * L[0, 1]) + f * L[0, 1] * ec * (e3 - ec)
        tmp1 = volume * num_p / den_p

        num_v = (ec - e1) * (ec + (e3 - ec) * (L[1, 0] - f * L[1, 1])) + f * ec * (e3 - ec)
        den_v = (ec + (e3 - ec) * (L[1, 0] - f * L[1, 1])) * (e1 + (ec - e1) * L[1, 1]) + f * L[1, 1] * ec * (e3 - ec)
        tmp2 = volume * num_v / den_v

        alpha_all[i] = _apply_mpole_order(mpole_order, tmp1, tmp2, 0.0, 0.0, imgs, d)
    return alpha_all


def yamaguchi_polarizabilities(state: SpheroidInitState, *, nint: int = 250) -> np.ndarray:
    """Yamaguchi(): dispatch Spheroid_Yamaguchi or Coating_Yamaguchi by geometry."""
    geom = state.case.geometry.strip()
    if geom.lower() == "yamaguchi":
        return spheroid_yamaguchi(state)
    if geom.lower() == "coated":
        return coating_yamaguchi(state, nint=nint)
    raise ValueError(f"unsupported Yamaguchi geometry: {geom!r}")
