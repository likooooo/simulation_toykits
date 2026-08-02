"""Step 1b: geometry integrals (integral_mod.f90 get_integrals_gauleg)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from granfilm.common.legendre import ass_legendre, deriv_ass_legendre, gauss_legendre


@dataclass
class IntegralsVolume:
    """Int(l1,l2,m) with Q, K, L, M, N each mu in {0,1} -> index 0,1."""

    Q: np.ndarray  # (L+1, L+1, M+1)
    K: np.ndarray  # (L+1, L+1, M+1, 2)
    L: np.ndarray
    M: np.ndarray
    N: np.ndarray

    @classmethod
    def zeros(cls, mpole_order: int, m_max: int) -> IntegralsVolume:
        sh = (mpole_order + 1, mpole_order + 1, m_max + 1)
        sh4 = sh + (2,)
        return cls(
            Q=np.zeros(sh, dtype=np.float64),
            K=np.zeros(sh4, dtype=np.float64),
            L=np.zeros(sh4, dtype=np.float64),
            M=np.zeros(sh4, dtype=np.float64),
            N=np.zeros(sh4, dtype=np.float64),
        )


def _mu_func(tr: float, kind: int, mupos: float) -> float:
    if kind == 1:
        return mupos
    if kind == 2:
        return 2.0 * tr - mupos
    raise ValueError(f"mu_func kind={kind}")


def _gamma_func(x: np.ndarray, mu: float) -> np.ndarray:
    return 1.0 - 2.0 * x * mu + mu**2


def _chi_func(x: np.ndarray, mu: float) -> np.ndarray:
    return (x - mu) / np.sqrt(_gamma_func(x, mu))


def step1_integrals(
    tr: float,
    MPpos: float,
    mpole_order: int,
    m_max: int,
    *,
    nint: int = 250,
) -> IntegralsVolume:
    """Compute Int for given truncation tr (also used with tr=1 for int_tr1)."""
    mpo = mpole_order
    mm = m_max
    Int = IntegralsVolume.zeros(mpo, mm)

    x, w = gauss_legendre(-1.0, tr, nint)
    mu = np.array([_mu_func(tr, 1, MPpos), _mu_func(tr, 2, MPpos)], dtype=np.float64)
    gamma = np.stack([_gamma_func(x, mu[0]), _gamma_func(x, mu[1])], axis=0)
    chi = np.stack([_chi_func(x, mu[0]), _chi_func(x, mu[1])], axis=0)

    # Precompute polynomials poly[l,m,:] for m=0,1 and optionally m=2
    poly_p: dict[tuple[int, int], np.ndarray] = {}
    poly_pp: dict[tuple[int, int, int], np.ndarray] = {}
    poly_dp: dict[tuple[int, int, int], np.ndarray] = {}

    poly_p[(0, 0)] = np.ones_like(x)
    for imu in range(2):
        poly_pp[(0, 0, imu)] = np.zeros_like(chi[imu])
        poly_dp[(0, 0, imu)] = np.zeros_like(chi[imu])
    for l in range(1, mpo + 1):
        for m in range(0, 2):
            poly_p[(l, m)] = ass_legendre(l, m, x)
            for imu in range(2):
                poly_pp[(l, m, imu)] = ass_legendre(l, m, chi[imu])
                poly_dp[(l, m, imu)] = deriv_ass_legendre(l, m, chi[imu])

    if mm == 2:
        for l in range(2, mpo + 1):
            poly_p[(l, 2)] = ass_legendre(l, 2, x)
            for imu in range(2):
                poly_pp[(l, 2, imu)] = ass_legendre(l, 2, chi[imu])
                poly_dp[(l, 2, imu)] = deriv_ass_legendre(l, 2, chi[imu])

    Int.Q[0, 0, 0] = tr + 1.0

    for l2 in range(mpo + 1):
        integrand = poly_p[(l2, 0)]
        Int.Q[0, l2, 0] = float(np.sum(w * integrand))
        for imu in range(2):
            g = gamma[imu]
            Int.K[0, l2, 0, imu] = float(
                np.sum(w * poly_pp[(l2, 0, imu)] * g ** (-(l2 + 1) / 2.0))
            )
            Int.L[0, l2, 0, imu] = float(
                np.sum(
                    w
                    * (
                        -(l2 + 1) * g ** (-0.5 * l2 - 1.5) * (1.0 - mu[imu] * x) * poly_pp[(l2, 0, imu)]
                        + g ** (-0.5 * l2 - 2.0)
                        * (x * g - (x - mu[imu]) * (1.0 - mu[imu] * x))
                        * poly_dp[(l2, 0, imu)]
                    )
                )
            )
            Int.M[0, l2, 0, imu] = float(np.sum(w * poly_pp[(l2, 0, imu)] * g ** (l2 / 2.0)))
            Int.N[0, l2, 0, imu] = float(
                np.sum(
                    w
                    * (
                        l2 * g ** (0.5 * l2 - 1.0) * (1.0 - mu[imu] * x) * poly_pp[(l2, 0, imu)]
                        + g ** (0.5 * l2 - 1.5)
                        * (x * g - (x - mu[imu]) * (1.0 - mu[imu] * x))
                        * poly_dp[(l2, 0, imu)]
                    )
                )
            )

    for l1 in range(1, mpo + 1):
        integrand = poly_p[(l1, 0)]
        Int.Q[l1, 0, 0] = float(np.sum(w * integrand))
        for imu in range(2):
            g = gamma[imu]
            Int.K[l1, 0, 0, imu] = float(np.sum(w * poly_p[(l1, 0)] * g ** (-0.5)))
            Int.L[l1, 0, 0, imu] = float(
                np.sum(
                    w
                    * poly_p[(l1, 0)]
                    * (
                        -g ** (-1.5) * (1.0 - mu[imu] * x)
                        + g ** (-2.0) * (x * g - (x - mu[imu]) * (1.0 - mu[imu] * x))
                    )
                )
            )
            Int.M[l1, 0, 0, imu] = float(np.sum(w * poly_p[(l1, 0)]))
            Int.N[l1, 0, 0, imu] = float(
                np.sum(
                    w
                    * poly_p[(l1, 0)]
                    * (g ** (-1.5) * (x * g - (x - mu[imu]) * (1.0 - mu[imu] * x)))
                )
            )

    for l1 in range(1, mpo + 1):
        for l2 in range(1, mpo + 1):
            for m in range(0, 2):
                integrand = poly_p[(l1, m)] * poly_p[(l2, m)]
                Int.Q[l1, l2, m] = float(np.sum(w * integrand))
                for imu in range(2):
                    g = gamma[imu]
                    Int.K[l1, l2, m, imu] = float(
                        np.sum(w * poly_p[(l1, m)] * poly_pp[(l2, m, imu)] * g ** (-(l2 + 1) / 2.0))
                    )
                    Int.L[l1, l2, m, imu] = float(
                        np.sum(
                            w
                            * poly_p[(l1, m)]
                            * (
                                -(l2 + 1)
                                * g ** (-0.5 * l2 - 1.5)
                                * (1.0 - mu[imu] * x)
                                * poly_pp[(l2, m, imu)]
                                + g ** (-0.5 * l2 - 2.0)
                                * (x * g - (x - mu[imu]) * (1.0 - mu[imu] * x))
                                * poly_dp[(l2, m, imu)]
                            )
                        )
                    )
                    Int.M[l1, l2, m, imu] = float(
                        np.sum(w * poly_p[(l1, m)] * poly_pp[(l2, m, imu)] * g ** (l2 / 2.0))
                    )
                    Int.N[l1, l2, m, imu] = float(
                        np.sum(
                            w
                            * poly_p[(l1, m)]
                            * (
                                l2 * g ** (0.5 * l2 - 1.0) * (1.0 - mu[imu] * x) * poly_pp[(l2, m, imu)]
                                + g ** (0.5 * l2 - 1.5)
                                * (x * g - (x - mu[imu]) * (1.0 - mu[imu] * x))
                                * poly_dp[(l2, m, imu)]
                            )
                        )
                    )

    if mm == 2:
        for l1 in range(2, mpo + 1):
            for l2 in range(2, mpo + 1):
                m = 2
                integrand = poly_p[(l1, 2)] * poly_p[(l2, 2)]
                Int.Q[l1, l2, m] = float(np.sum(w * integrand))
                for imu in range(2):
                    g = gamma[imu]
                    Int.K[l1, l2, m, imu] = float(
                        np.sum(w * poly_p[(l1, 2)] * poly_pp[(l2, 2, imu)] * g ** (-(l2 + 1) / 2.0))
                    )
                    Int.L[l1, l2, m, imu] = float(
                        np.sum(
                            w
                            * poly_p[(l1, 2)]
                            * (
                                -(l2 + 1)
                                * g ** (-0.5 * l2 - 1.5)
                                * (1.0 - mu[imu] * x)
                                * poly_pp[(l2, 2, imu)]
                                + g ** (-0.5 * l2 - 2.0)
                                * (x * g - (x - mu[imu]) * (1.0 - mu[imu] * x))
                                * poly_dp[(l2, 2, imu)]
                            )
                        )
                    )
                    Int.M[l1, l2, m, imu] = float(
                        np.sum(w * poly_p[(l1, 2)] * poly_pp[(l2, 2, imu)] * g ** (l2 / 2.0))
                    )
                    Int.N[l1, l2, m, imu] = float(
                        np.sum(
                            w
                            * poly_p[(l1, 2)]
                            * (
                                l2 * g ** (0.5 * l2 - 1.0) * (1.0 - mu[imu] * x) * poly_pp[(l2, 2, imu)]
                                + g ** (0.5 * l2 - 1.5)
                                * (x * g - (x - mu[imu]) * (1.0 - mu[imu] * x))
                                * poly_dp[(l2, 2, imu)]
                            )
                        )
                    )

    return Int
