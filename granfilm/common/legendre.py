"""Associated Legendre polynomials and Gauss-Legendre quadrature (GranFilm legendre.f90)."""

from __future__ import annotations

import math

import numpy as np

from granfilm.common.constants import PI


def arth(first: float, increment: float, n: int) -> np.ndarray:
    return first + increment * np.arange(n, dtype=np.float64)


def gauss_legendre(x1: float, x2: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre quadrature on [x1, x2] (NR gauleg_dp, matches GranFilm legendre.f90)."""
    eps = 3.0e-14
    x = np.empty(n, dtype=np.float64)
    w = np.empty(n, dtype=np.float64)
    m = (n + 1) // 2
    xm = 0.5 * (x2 + x1)
    xl = 0.5 * (x2 - x1)
    z = np.cos(np.pi * (arth(1.0, 1.0, m) - 0.25) / (n + 0.5))
    unfinished = np.ones(m, dtype=bool)
    for _its in range(10):
        if not np.any(unfinished):
            break
        p1 = np.ones(m, dtype=np.float64)
        p2 = np.zeros(m, dtype=np.float64)
        for j in range(1, n + 1):
            if not np.any(unfinished):
                break
            p3 = p2.copy()
            p2 = p1.copy()
            p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
        pp = n * (z * p1 - p2) / (z * z - 1.0)
        z1 = z.copy()
        z = z1 - p1 / pp
        unfinished = np.abs(z - z1) > eps
    x[:m] = xm - xl * z
    x[n - m :][::-1] = xm + xl * z
    w[:m] = 2.0 * xl / ((1.0 - z * z) * pp * pp)
    w[n - m :][::-1] = w[:m]
    return x, w


def plgndr(l: int, m: int, x: float | np.ndarray) -> np.ndarray | float:
    """Scalar/vector associated Legendre P_l^m(x), NR plgndr_s_dp convention."""
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    out = np.empty_like(x_arr)
    for idx, xv in enumerate(x_arr):
        if not (0 <= m <= l and abs(xv) <= 1.0):
            raise ValueError(f"bad plgndr args l={l} m={m} x={xv}")
        pmm = 1.0
        if m > 0:
            somx2 = math.sqrt((1.0 - xv) * (1.0 + xv))
            pmm = float(np.prod(arth(1.0, 2.0, m))) * somx2**m
            if m % 2 == 1:
                pmm = -pmm
        if l == m:
            out[idx] = pmm
        else:
            pmmp1 = xv * (2 * m + 1) * pmm
            if l == m + 1:
                out[idx] = pmmp1
            else:
                pll = 0.0
                for ll in range(m + 2, l + 1):
                    pll = (xv * (2 * ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
                    pmm = pmmp1
                    pmmp1 = pll
                out[idx] = pll
    if np.ndim(x) == 0:
        return float(out[0])
    return out


def ass_legendre(l: int, m: int, x: np.ndarray) -> np.ndarray:
    return np.asarray(plgndr(l, m, x), dtype=np.float64)


def deriv_ass_legendre(l: int, m: int, x: np.ndarray) -> np.ndarray:
    p_lp1 = ass_legendre(l + 1, m, x)
    p_lm = ass_legendre(l, m, x)
    return ((l - m + 1) * p_lp1 - (l + 1) * x * p_lm) / (x**2 - 1.0)
