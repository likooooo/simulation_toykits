"""Fortran linsolver_mod port: iterative refinement (zgerfs-style)."""

from __future__ import annotations

import numpy as np


def _relative_residual(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    r = b - A @ x
    bn = float(np.linalg.norm(b))
    if bn == 0.0:
        return float(np.linalg.norm(r))
    return float(np.linalg.norm(r) / bn)


def linsolve_granfilm(A: np.ndarray, b: np.ndarray, epslin: float = 1e-4) -> np.ndarray:
    """
    Match GranFilm linsolver intent: refine until rel residual < epslin.
    (Fortran also falls back to FMZM multiprecision when this is insufficient.)
    """
    A = np.asarray(A, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    x = np.linalg.solve(A, b)
    if _relative_residual(A, x, b) < epslin:
        return x
    for _ in range(12):
        r = b - A @ x
        x = x + np.linalg.solve(A, r)
        if _relative_residual(A, x, b) < epslin:
            return x
    return x
