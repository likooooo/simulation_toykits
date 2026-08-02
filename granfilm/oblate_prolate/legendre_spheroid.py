"""Legendre functions on imaginary / real spheroidal axes (oblate_int_gauleg / prolate_int_gauleg)."""

from __future__ import annotations

import numpy as np


def xfactor_func(l: int, m: int) -> float:
    factor = 1.0
    if m == 1 and l != 0:
        factor /= l
    for i in range(1, l + 1):
        factor *= i / (2 * i - 1)
    return factor


def zfactor_func(l: int, m: int) -> float:
    factor = 1.0
    if m == 1:
        factor /= l + 1
    for i in range(1, l + 1):
        factor *= (2 * i + 1) / i
    return factor


def legendre_p_aimag(x: np.ndarray, lmax: int) -> tuple[np.ndarray, np.ndarray]:
    """P(l,m,ix) and dP/dx for purely imaginary argument (oblate)."""
    npts = x.size
    p = np.zeros((npts, lmax + 1, 2), dtype=np.complex128)
    dp = np.zeros((npts, lmax + 1, 2), dtype=np.complex128)
    for ix, y in enumerate(x):
        z = 1.0 + y**2
        for m in (0, 1):
            if m == 0:
                p[ix, 0, 0] = 1.0
                p[ix, 1, 0] = 1j * y
                dp[ix, 0, 0] = 0.0
            else:
                p[ix, 0, 1] = 0.0
                dp[ix, 0, 1] = 0.0
                p[ix, 1, 1] = np.sqrt(z)
                if lmax > 1:
                    p[ix, 2, 1] = 3j * y * np.sqrt(z)
                dp[ix, 1, 1] = y / np.sqrt(z)
            for l in range(1 + m, lmax):
                p[ix, l + 1, m] = ((2 * l + 1) * 1j * y * p[ix, l, m] - (l + m) * p[ix, l - 1, m]) / (l - m + 1)
                dp[ix, l, m] = (l * y * p[ix, l, m] + 1j * (l + m) * p[ix, l - 1, m]) / z
            l = lmax
            if l >= m + 1:
                dp[ix, l, m] = (l * y * p[ix, l, m] + 1j * (l + m) * p[ix, l - 1, m]) / z
    return p, dp


def legendre_q_aimag(x: np.ndarray, lmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Q(l,m,ix) and dQ/dx for purely imaginary argument (oblate)."""
    npts = x.size
    q = np.zeros((npts, lmax + 1, 2), dtype=np.complex128)
    dq = np.zeros((npts, lmax + 1, 2), dtype=np.complex128)
    for ix, y in enumerate(x):
        z = 1.0 + y**2
        w = np.arctan(1.0 / y)
        for m in (0, 1):
            if m == 0:
                q[ix, 0, 0] = -1j * w
                q[ix, 1, 0] = y * w - 1.0
                dq[ix, 0, 0] = 1j / z
            else:
                q[ix, 0, 1] = 0.0
                dq[ix, 0, 1] = 0.0
                q[ix, 1, 1] = -np.sqrt(z) * (w - y / z)
                if lmax > 1:
                    q[ix, 2, 1] = 1j * np.sqrt(z) * ((2 + 3 * y**2) / z - 3 * y * w)
                dq[ix, 1, 1] = (-y * w + (2 + y**2) / z) / np.sqrt(z)
            for l in range(1 + m, lmax):
                q[ix, l + 1, m] = ((2 * l + 1) * 1j * y * q[ix, l, m] - (l + m) * q[ix, l - 1, m]) / (l - m + 1)
                dq[ix, l, m] = (l * y * q[ix, l, m] + 1j * (l + m) * q[ix, l - 1, m]) / z
            l = lmax
            if l >= m + 1:
                dq[ix, l, m] = (l * y * q[ix, l, m] + 1j * (l + m) * q[ix, l - 1, m]) / z
    return q, dq


def legendre_p_real(x: np.ndarray, lmax: int) -> tuple[np.ndarray, np.ndarray]:
    """P(l,m,ix) for real argument xi > 1 (prolate)."""
    npts = x.size
    p = np.zeros((npts, lmax + 1, 2), dtype=np.complex128)
    dp = np.zeros((npts, lmax + 1, 2), dtype=np.complex128)
    for ix, y in enumerate(x):
        z = y**2 - 1.0
        for m in (0, 1):
            if m == 0:
                p[ix, 0, 0] = 1.0
                p[ix, 1, 0] = y
                dp[ix, 0, 0] = 0.0
            else:
                p[ix, 0, 1] = 0.0
                dp[ix, 0, 1] = 0.0
                p[ix, 1, 1] = -1j * np.sqrt(z)
                if lmax > 1:
                    p[ix, 2, 1] = -1j * 3 * y * np.sqrt(z)
                dp[ix, 1, 1] = -1j * y / np.sqrt(z)
            for l in range(1 + m, lmax):
                p[ix, l + 1, m] = ((2 * l + 1) * y * p[ix, l, m] - (l + m) * p[ix, l - 1, m]) / (l - m + 1)
                dp[ix, l, m] = (l * y * p[ix, l, m] - (l + m) * p[ix, l - 1, m]) / z
            l = lmax
            if l >= m + 1:
                dp[ix, l, m] = (l * y * p[ix, l, m] - (l + m) * p[ix, l - 1, m]) / z
    return p, dp


def legendre_q_real(x: np.ndarray, lmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Q(l,m,ix) for real argument xi > 1 (prolate)."""
    npts = x.size
    q = np.zeros((npts, lmax + 1, 2), dtype=np.complex128)
    dq = np.zeros((npts, lmax + 1, 2), dtype=np.complex128)
    for ix, y in enumerate(x):
        z = y**2 - 1.0
        w = 0.5 * np.log((y + 1.0) / (y - 1.0))
        for m in (0, 1):
            if m == 0:
                q[ix, 0, 0] = w
                q[ix, 1, 0] = y * w - 1.0
                dq[ix, 0, 0] = -1.0 / z
            else:
                q[ix, 0, 1] = 0.0
                dq[ix, 0, 1] = 0.0
                q[ix, 1, 1] = -w * np.sqrt(z) + y / np.sqrt(z)
                if lmax > 1:
                    q[ix, 2, 1] = -3 * y * np.sqrt(z) * w + (3 * y**2 - 2) / np.sqrt(z)
                dq[ix, 1, 1] = -(w * y + (2 - y**2) / z) / np.sqrt(z)
            for l in range(1 + m, lmax):
                q[ix, l + 1, m] = ((2 * l + 1) * y * q[ix, l, m] - (l + m) * q[ix, l - 1, m]) / (l - m + 1)
                dq[ix, l, m] = (l * y * q[ix, l, m] - (l + m) * q[ix, l - 1, m]) / z
            l = lmax
            if l >= m + 1:
                dq[ix, l, m] = (l * y * q[ix, l, m] - (l + m) * q[ix, l - 1, m]) / z
    return q, dq
