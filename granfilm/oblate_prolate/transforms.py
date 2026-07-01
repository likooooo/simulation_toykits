"""Coordinate transforms for oblate/prolate image integrals."""

from __future__ import annotations

import numpy as np


def xi_eta_trans_oblate(xi: float, eta: np.ndarray, xi1: float) -> tuple[np.ndarray, np.ndarray]:
    r = xi1 / xi
    xi_trans = np.empty_like(eta)
    eta_trans = np.empty_like(eta)
    for i, e in enumerate(eta):
        tmp = 1.0 + 4 * r**2 - 4 * r * e - (e / xi) ** 2
        tmp = tmp + np.sqrt(tmp**2 + (2 * (2 * r - e) / xi) ** 2)
        tmp = np.sqrt(tmp)
        xi_trans[i] = xi * tmp / np.sqrt(2.0)
        eta_trans[i] = np.sqrt(2.0) * (e - 2 * r) / tmp
    return xi_trans, eta_trans


def deriv_xi_eta_trans_oblate(
    xi: float, eta: np.ndarray, xi1: float
) -> tuple[np.ndarray, np.ndarray]:
    r = xi1 / xi
    deriv_xi = np.empty_like(eta)
    deriv_eta = np.empty_like(eta)
    for i, e in enumerate(eta):
        t1 = 1.0 + 4 * r**2 - 4 * r * e - (e / xi) ** 2
        t2 = 2 * (2 * r - e) / xi
        t3 = -8 * r**3 / xi1 + 4 * r**2 * e / xi1 + 2 * e**2 / xi**3
        t4 = -8 * xi1 / xi**3 + 2 * e / xi**2
        t5 = np.sqrt(t1**2 + t2**2)
        t6 = t1 * t3 + t2 * t4
        t7 = np.sqrt(t1 + t5)
        t8 = (t3 + t6 / t5) / (2 * t7)
        t9 = t7**3
        t10 = (t3 + t6 / t5) / (2 * t9)
        deriv_xi[i] = (t7 + xi * t8) / np.sqrt(2.0)
        deriv_eta[i] = (2 * xi1 / xi**2 / t7 - (e - 2 * r) * t10) * np.sqrt(2.0)
    return deriv_xi, deriv_eta


def xi_eta_trans_prolate(xi: float, eta: np.ndarray, doa: float) -> tuple[np.ndarray, np.ndarray]:
    r = doa / xi
    xi_trans = np.empty_like(eta)
    eta_trans = np.empty_like(eta)
    for i, e in enumerate(eta):
        tmp = 1.0 + 4 * r**2 - 4 * r * e + (e / xi) ** 2
        tmp = tmp + np.sqrt(tmp**2 - (2 * (2 * r - e) / xi) ** 2)
        tmp = np.sqrt(tmp)
        xi_trans[i] = xi * tmp / np.sqrt(2.0)
        eta_trans[i] = np.sqrt(2.0) * (e - 2 * r) / tmp
    return xi_trans, eta_trans


def deriv_xi_eta_trans_prolate(
    xi: float, eta: np.ndarray, doa: float
) -> tuple[np.ndarray, np.ndarray]:
    r = doa / xi
    deriv_xi = np.empty_like(eta)
    deriv_eta = np.empty_like(eta)
    for i, e in enumerate(eta):
        t1 = 1.0 + 4 * r**2 - 4 * r * e + (e / xi) ** 2
        t2 = 2 * (2 * r - e) / xi
        t3 = -8 * r**3 / doa + 4 * r**2 * e / doa - 2 * e**2 / xi**3
        t4 = -8 * doa / xi**3 + 2 * e / xi**2
        t5 = np.sqrt(t1**2 - t2**2)
        t6 = t1 * t3 - t2 * t4
        t7 = np.sqrt(t1 + t5)
        t8 = (t3 + t6 / t5) / (2 * t7)
        t9 = t7**3
        t10 = (t3 + t6 / t5) / (2 * t9)
        deriv_xi[i] = (t7 + xi * t8) / np.sqrt(2.0)
        deriv_eta[i] = (2 * doa / xi**2 / t7 - (e - 2 * r) * t10) * np.sqrt(2.0)
    return deriv_xi, deriv_eta
