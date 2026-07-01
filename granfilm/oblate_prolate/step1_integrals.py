"""Step 1: spheroidal integrals Q,V,W,dVdx,dWdx (Gauleg)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from granfilm.common.legendre import ass_legendre, deriv_ass_legendre, gauss_legendre
from granfilm.oblate_prolate.step0_init import SpheroidInitState
from granfilm.oblate_prolate.transforms import (
    deriv_xi_eta_trans_oblate,
    deriv_xi_eta_trans_prolate,
    xi_eta_trans_oblate,
    xi_eta_trans_prolate,
)
from granfilm.oblate_prolate.xz import xz_oblate, xz_prolate


def _legendre_grid(x: np.ndarray, mpo: int) -> tuple[np.ndarray, np.ndarray]:
    n = x.size
    p = np.zeros((n, mpo + 1, 2), dtype=np.float64)
    dp = np.zeros((n, mpo + 1, 2), dtype=np.float64)
    for m in (0, 1):
        for l in range(mpo + 1):
            if l < m:
                continue
            p[:, l, m] = ass_legendre(l, m, x)
            dp[:, l, m] = deriv_ass_legendre(l, m, x)
    return p, dp


@dataclass
class SpheroidIntegrals:
    Q: np.ndarray
    V: np.ndarray
    W: np.ndarray
    dVdx: np.ndarray
    dWdx: np.ndarray


def _integrals_oblate(state: SpheroidInitState, *, nint: int) -> SpheroidIntegrals:
    mpo = state.case.Mpole_order
    tr = state.case.tr if state.above else abs(state.case.tr)
    xi0 = state.xi0
    xi1 = tr * xi0
    eta, w = gauss_legendre(-1.0, tr, nint)
    xi_trans, eta_trans = xi_eta_trans_oblate(xi0, eta, xi1)
    dxi, deta = deriv_xi_eta_trans_oblate(xi0, eta, xi1)
    xz = xz_oblate(xi_trans, mpo)
    p, dp = _legendre_grid(eta, mpo)
    p_t, dp_t = _legendre_grid(eta_trans, mpo)
    return _accumulate_integrals(mpo, w, p, p_t, dp, dp_t, dxi, deta, xz)


def _integrals_prolate(state: SpheroidInitState, *, nint: int) -> SpheroidIntegrals:
    mpo = state.case.Mpole_order
    tr = state.case.tr if state.above else abs(state.case.tr)
    xi0 = state.xi0
    doa = abs(state.div_surface) / state.a
    eta, w = gauss_legendre(-1.0, tr, nint)
    xi_trans, eta_trans = xi_eta_trans_prolate(xi0, eta, doa)
    dxi, deta = deriv_xi_eta_trans_prolate(xi0, eta, doa)
    xz = xz_prolate(xi_trans, mpo)
    p, dp = _legendre_grid(eta, mpo)
    p_t, dp_t = _legendre_grid(eta_trans, mpo)
    return _accumulate_integrals(mpo, w, p, p_t, dp, dp_t, dxi, deta, xz)


def _accumulate_integrals(
    mpo: int,
    w: np.ndarray,
    p: np.ndarray,
    p_t: np.ndarray,
    dp: np.ndarray,
    dp_t: np.ndarray,
    dxi: np.ndarray,
    deta: np.ndarray,
    xz,
) -> SpheroidIntegrals:
    q = np.zeros((mpo + 1, mpo + 1, 2), dtype=np.float64)
    v = np.zeros_like(q)
    ww = np.zeros_like(q)
    dv = np.zeros_like(q)
    dw = np.zeros_like(q)
    for l1 in range(mpo + 1):
        for l2 in range(mpo + 1):
            for m in (0, 1):
                q[l1, l2, m] = np.sum(w * p[:, l1, m] * p[:, l2, m])
                v[l1, l2, m] = np.sum(w * p[:, l1, m] * p_t[:, l2, m] * xz.Z[:, l2, m])
                ww[l1, l2, m] = np.sum(w * p[:, l1, m] * p_t[:, l2, m] * xz.X[:, l2, m])
                dv[l1, l2, m] = np.sum(
                    w
                    * p[:, l1, m]
                    * (
                        deta * dp_t[:, l2, m] * xz.Z[:, l2, m]
                        + dxi * p_t[:, l2, m] * xz.dZdxi[:, l2, m]
                    )
                )
                dw[l1, l2, m] = np.sum(
                    w
                    * p[:, l1, m]
                    * (
                        deta * dp_t[:, l2, m] * xz.X[:, l2, m]
                        + dxi * p_t[:, l2, m] * xz.dXdxi[:, l2, m]
                    )
                )
    return SpheroidIntegrals(Q=q, V=v, W=ww, dVdx=dv, dWdx=dw)


def step1_integrals(state: SpheroidInitState, *, nint: int | None = None) -> SpheroidIntegrals:
    n = nint if nint is not None else state.case.Nint
    if state.island_type == "oblate":
        return _integrals_oblate(state, nint=n)
    if state.island_type == "prolate":
        return _integrals_prolate(state, nint=n)
    raise ValueError(f"unsupported island_type={state.island_type!r}")


def step1_xz_at_xi0(state: SpheroidInitState) -> np.ndarray:
    """Return XZ at xi0 as array [l,m] fields (X, dXdxi, Z, dZdxi)."""
    mpo = state.case.Mpole_order
    if state.island_type == "oblate":
        xz = xz_oblate(state.xi0, mpo)
    else:
        xz = xz_prolate(state.xi0, mpo)
    return xz
