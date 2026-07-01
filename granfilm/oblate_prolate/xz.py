"""XZ spheroidal basis functions at xi0."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from granfilm.oblate_prolate.legendre_spheroid import (
    legendre_p_aimag,
    legendre_p_real,
    legendre_q_aimag,
    legendre_q_real,
    xfactor_func,
    zfactor_func,
)


@dataclass
class XZField:
    X: np.ndarray
    dXdxi: np.ndarray
    Z: np.ndarray
    dZdxi: np.ndarray


def xz_oblate(xi: np.ndarray | float, mpo: int) -> XZField:
    xi_arr = np.atleast_1d(np.asarray(xi, dtype=np.float64))
    npts = xi_arr.size
    p, dp = legendre_p_aimag(xi_arr, mpo)
    q, dq = legendre_q_aimag(xi_arr, mpo)
    x = np.zeros((npts, mpo + 1, 2), dtype=np.float64)
    dx = np.zeros_like(x)
    z = np.zeros_like(x)
    dz = np.zeros_like(x)
    for m in (0, 1):
        for l in range(mpo + 1):
            xf = (1j ** (m - l)) * xfactor_func(l, m)
            zf = (1j ** (l + 1)) * zfactor_func(l, m)
            x[:, l, m] = np.real(xf * p[:, l, m])
            dx[:, l, m] = np.real(xf * dp[:, l, m])
            z[:, l, m] = np.real(zf * q[:, l, m])
            dz[:, l, m] = np.real(zf * dq[:, l, m])
    if np.isscalar(xi):
        return XZField(x[0], dx[0], z[0], dz[0])
    return XZField(x, dx, z, dz)


def xz_prolate(xi: np.ndarray | float, mpo: int) -> XZField:
    xi_arr = np.atleast_1d(np.asarray(xi, dtype=np.float64))
    npts = xi_arr.size
    p, dp = legendre_p_real(xi_arr, mpo)
    q, dq = legendre_q_real(xi_arr, mpo)
    x = np.zeros((npts, mpo + 1, 2), dtype=np.float64)
    dx = np.zeros_like(x)
    z = np.zeros_like(x)
    dz = np.zeros_like(x)
    for m in (0, 1):
        for l in range(mpo + 1):
            xf = (1j**m) * xfactor_func(l, m)
            zf = zfactor_func(l, m)
            x[:, l, m] = np.real(xf * p[:, l, m])
            dx[:, l, m] = np.real(xf * dp[:, l, m])
            z[:, l, m] = np.real(zf * q[:, l, m])
            dz[:, l, m] = np.real(zf * dq[:, l, m])
    if np.isscalar(xi):
        return XZField(x[0], dx[0], z[0], dz[0])
    return XZField(x, dx, z, dz)
