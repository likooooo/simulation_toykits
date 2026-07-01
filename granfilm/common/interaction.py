"""Island–island lattice sums (GranFilm interaction_mod.f90)."""

from __future__ import annotations

import numpy as np

from granfilm.common.constants import PI
from granfilm.common.legendre import gauss_legendre


def argument(l: int, d: float, r: float) -> float:
    """Spherical harmonic kernel Argument(l,d,r) in interaction_mod.f90."""
    cosi = d / r
    if l == 2:
        return (3 * cosi**2 - 1.0) / 2.0 / r**3
    if l == 3:
        return (5 * cosi**3 - 3 * cosi) / 2.0 / r**4
    if l == 4:
        return (35 * cosi**4 - 30 * cosi**2 + 3.0) / 8.0 / r**5
    raise ValueError(f"Argument l={l} not supported")


def pair_correlation(x: np.ndarray) -> np.ndarray:
    """Pair_Correlation(x) — RPT pair correlation (Fortran: g = 1)."""
    return np.ones_like(x, dtype=np.float64)


def integral_rpt(d: float, lmax: float, rapp: float, nint: int) -> np.ndarray:
    """Integral_RPT(d,Lmax,Rapp,Integral,Nint) — Barrera RPT integrals Fxx,Fxz,Fzz."""
    x, w = gauss_legendre(1.0, lmax, nint)
    y = pair_correlation(2.0 * rapp * x)
    r = np.sqrt(x**2 + d**2)
    integrals = np.empty(6, dtype=np.float64)
    integrals[0] = np.sum(w * y / x**5)
    integrals[1] = np.sum(w * y * (x**2 - d**2 / 5.0) / x**2 / r**5)
    integrals[2] = np.sum(w * y * x * (x**4 - 2.0 * (d * x) ** 2 / 5.0 + 2.0 * d**4 / 5.0) / r**10)
    integrals[3] = np.sum(w * y * (x**2 - 2.0 * d**2) / x**2 / r**5)
    integrals[4] = np.sum(w * y * x * (x**4 - 4.0 * (d * x) ** 2 + 4.0 * d**4) / r**10)
    integrals[5] = np.sum(w * y * d**2 * x**3 / r**10)
    return integrals


def renorm_polarizability(
    d_mu: float,
    alpha_dipole: np.ndarray,
    *,
    eps_vacuum: complex,
    eps_substrate: complex,
    coverage: float,
    R: float,
    Rapparent: float,
    levels: int,
    density: float,
    above: bool,
    nint: int = 250,
) -> None:
    """Renorm_Polarizability(d_mu,param,alpha) — mutates alpha_dipole[0:2] in place."""
    epsilon_tol = 1.0e-6
    nitmax = 35

    e1 = eps_vacuum
    phi = coverage
    norm = R / Rapparent

    alpha_dipole[:] = alpha_dipole * norm
    d = d_mu * norm
    lmax = levels / np.sqrt(density) / (2.0 * Rapparent)
    integral = integral_rpt(d, lmax, Rapparent, nint)

    e2 = eps_substrate
    if above:
        alpha_dim = alpha_dipole / (4.0 * PI * e1)
        eps = (e2 - e1) / (e2 + e1)
    else:
        alpha_dim = alpha_dipole / (4.0 * PI * e2)
        eps = (e1 - e2) / (e2 + e1)

    fxx = 5.0 * phi * (integral[0] - 2.0 * eps * integral[1] + eps**2 * integral[2]) / 4.0
    fzz = phi * (integral[0] + 2.0 * eps * integral[4] + eps**2 * integral[5]) / 2.0
    fxz = -9.0 * phi * (eps**2 * integral[5]) / 16.0

    pol_old = alpha_dim.copy()
    for nit in range(1, nitmax + 1):
        pol_new = np.empty(2, dtype=np.complex128)
        pol_new[0] = 1.0 + fxx / 4.0 * pol_old[0] ** 2 + fxz * pol_old[0] * pol_old[1]
        pol_new[1] = 1.0 + 2.0 * fxz * pol_old[0] * pol_old[1] + fzz / 4.0 * pol_old[1] ** 2
        pol_new *= alpha_dim
        error = np.abs((pol_new - pol_old) / pol_old)
        pol_old = pol_new
        if nit < nitmax and (error[0] >= epsilon_tol or error[1] >= epsilon_tol):
            continue
        break

    if above:
        alpha_dipole[:] = 4.0 * PI * e1 * pol_old / norm
    else:
        alpha_dipole[:] = 4.0 * PI * e2 * pol_old / norm


def ir_random(d: float, l: int, lmax: float, nint: int) -> float:
    """MFT/RPT Ir_random integral (Pair_Correlation = 1)."""
    r, w = gauss_legendre(1.0, lmax, nint)
    integrand = r * argument(l, d, np.sqrt(r**2 + d**2))
    return float(np.sum(w * integrand) * np.sqrt((2 * l + 1) / (4 * PI)))


def sr_square(d: float, l: int, level: int) -> float:
    """Sr_square(d,l,level) — square lattice sum."""
    sm = 0.0
    sd = 0.0
    s = 0.0
    for m in range(level, 0, -1):
        mf = float(m)
        sm += argument(l, d, np.sqrt(mf**2 + d**2))
        sd += argument(l, d, np.sqrt(2.0 * mf**2 + d**2))
        s = 0.0
        for n in range(m, -1, -1):
            nf = float(n)
            s += argument(l, d, np.sqrt(mf**2 + nf**2 + d**2))
    sr = 4.0 * (2.0 * s - sd - sm)
    return sr * np.sqrt((2 * l + 1) / (4 * PI))


def sr_hexagonal(d: float, l: int, level: int) -> float:
    """Sr_hexagonal(d,l,level) — hexagonal lattice sum."""
    sd = 0.0
    s = 0.0
    for m in range(level, 0, -1):
        mf = float(m)
        sd += argument(l, d, np.sqrt(3.0 * mf**2 + d**2))
    for m in range(level, 0, -1):
        mf = float(m)
        for n in range(-m + 1, m):
            nf = float(n)
            s += argument(l, d, np.sqrt(mf**2 + nf**2 + mf * nf + d**2))
    sr = 4.0 * (sd + s)
    return sr * np.sqrt((2 * l + 1) / (4 * PI))


def effective_lattice_L(
    *,
    network: str,
    lattice_const: float,
    R: float,
    density_dim: float,
) -> float:
    """Dimensionless L in surf_const_coef_dipole; density_dim = param%density*R^2."""
    net = network.strip().upper()
    if net in {"MFT", "RPT"}:
        return 1.0 / np.sqrt(density_dim)
    return lattice_const / R


def lattice_sum(
    d_mu: float,
    n: int,
    *,
    network: str,
    R: float,
    Rapparent: float,
    density: float,
    lattice_const: float,
    levels: int,
    nint: int = 250,
) -> float:
    """Port of interaction_mod.f90 lattice_sum."""
    net = network.strip().upper()
    if net == "SQUARE":
        d = 2.0 * d_mu * R / lattice_const
        return float(sr_square(d, n, levels))
    if net == "HEXAGONAL":
        d = 2.0 * d_mu * R / lattice_const
        return float(sr_hexagonal(d, n, levels))
    if net in {"MFT", "RPT"}:
        norm = 1.0 / np.sqrt(density) / (2.0 * Rapparent)
        d = 2.0 * d_mu * R / (2.0 * Rapparent)
        lmax = levels * norm
        return float(2 * PI * norm ** (n - 1) * ir_random(d, n, lmax, nint))
    raise NotImplementedError(f"lattice_sum for network={network!r} not ported")


def surf_const_quadrupole_values(
    alphad: np.ndarray,
    alphaq: np.ndarray,
    *,
    e1: complex,
    e2: complex,
    above: bool,
    d: float,
    density: float,
    L: float,
    s2_mp: float,
    s2_imp: float,
    s3_mp: float,
    s3_imp: float,
    s4_mp: float,
    s4_imp: float,
) -> tuple[complex, complex, complex, complex]:
    """
    Surface constitutive coefficients for quadrupole island interaction.
    Port of optics_mod.f90 surf_const_coef_quadrupole (Sphere and Spheroid).
    Returns (gamma, beta, tau, delta).
    """
    sq1 = np.sqrt(15.0 * PI / 7.0)
    sq2 = np.sqrt(PI)
    sq3 = np.sqrt(PI / 5.0)
    sq4 = np.sqrt(3.0 * PI / 35.0)
    sq5 = np.sqrt(PI / 35.0)
    sq6 = np.sqrt(5.0 * PI / 7.0)

    eps = (e1 - e2) / (e2 + e1)
    s2p = s2_mp + eps * s2_imp
    s2m = s2_mp - eps * s2_imp
    s3p = s3_mp + eps * s3_imp
    s3m = s3_mp - eps * s3_imp
    s4p = s4_mp + eps * s4_imp
    s4m = s4_mp - eps * s4_imp

    e_sub = e1 if above else e2
    a1010 = alphad[0, 1] / (4 * PI * e_sub)
    a1111 = alphad[0, 0] / (4 * PI * e_sub)
    a2111 = 3 * alphad[1, 0] / (4 * PI * e_sub * np.sqrt(5.0))
    a2010 = alphad[1, 1] / (2 * PI * e_sub * np.sqrt(5.0 / 3.0))
    a2121 = 3 * (alphaq[1] + 2 * alphaq[2]) / (4 * PI * e_sub)
    a2020 = (2 * alphaq[0] + 3 * alphaq[2]) / (2 * PI * e_sub)
    a1020 = 5.0 / 3.0 * a2010
    a1121 = 5.0 / 3.0 * a2111

    if above:
        dz = (
            (1.0 - 4 * a1010 * s2m / L**3 * sq3 - 6 * a1020 * s3m / L**4 * sq4)
            * (1.0 + 2 * a2010 * s3p / L**4 * sq1 + 4 * a2020 * s4p / L**5 * sq2)
            + (4 * a2010 * s2m / L**3 * sq3 + 6 * a2020 * s3m / L**4 * sq4)
            * (2 * a1010 * s3p / L**4 * sq1 + 4 * a1020 * s4p / L**5 * sq2)
        )
        dp = (
            (1.0 + 2 * a1111 * s2p / L**3 * sq3 + 6 * a1121 * s3p / L**4 * sq5)
            * (1.0 - 2 * a2111 * s3m / L**4 * sq6 - 8 * a2121 * s4m / L**5 * sq2 / 3)
            + (2 * a2111 * s2p / L**3 * sq3 + 6 * a2121 * s3p / L**4 * sq5)
            * (2 * a1111 * s3m / L**4 * sq6 + 8 * a1121 * s4m / L**5 * sq2 / 3)
        )
        az = 4 * PI * e1 / dz * (
            a1010 * (1.0 + 2 * a2010 * s3p / L**4 * sq1 + 4 * a2020 * s4p / L**5 * sq2)
            - a2010 * (2 * a1010 * s3p / L**4 * sq1 + 4 * a1020 * s4p / L**5 * sq2)
        )
        az10 = 2 * PI * e1 * np.sqrt(5.0 / 3.0) / dz * (
            a2010 * (1.0 - 4 * a1010 * s2m / L**3 * sq3 - 6 * a1020 * s3m / L**4 * sq4)
            + a1010 * (4 * a2010 * s2m / L**3 * sq3 + 6 * a2020 * s3m / L**4 * sq4)
        )
        ap = 4 * PI / dp * (
            a1111 * (1.0 - 2 * a2111 * s3m / L**4 * sq6 - 8 * a2121 * s4m / L**5 * sq2 / 3)
            + a2111 * (2 * a1111 * s3m / L**4 * sq6 + 8 * a1121 * s4m / L**5 * sq2 / 3)
        )
        ap10 = 4 * PI * e1 * np.sqrt(5.0) / 3 / dp * (
            a2111 * (1.0 + 2 * a1111 * s2p / L**3 * sq3 + 6 * a1121 * s3p / L**4 * sq5)
            - a1111 * (2 * a2111 * s2p / L**3 * sq3 + 6 * a2121 * s3p / L**4 * sq5)
        )
        return (
            density * ap,
            density * az / e1**2,
            -density * (ap10 - d * ap),
            -density * (az10 + ap10 - d * az - d * ap) / e1,
        )

    dz = (
        (1.0 - 4 * a1010 * s2p / L**3 * sq3 - 6 * a1020 * s3p / L**4 * sq4)
        * (1.0 + 2 * a2010 * s3m / L**4 * sq1 + 4 * a2020 * s4m / L**5 * sq2)
        + (4 * a2010 * s2p / L**3 * sq3 + 6 * a2020 * s3p / L**4 * sq4)
        * (2 * a1010 * s3m / L**4 * sq1 + 4 * a1020 * s4m / L**5 * sq2)
    )
    dp = (
        (1.0 + 2 * a1111 * s2m / L**3 * sq3 + 6 * a1121 * s3m / L**4 * sq5)
        * (1.0 - 2 * a2111 * s3p / L**4 * sq6 - 8 * a2121 * s4p / L**5 * sq2 / 3)
        + (2 * a2111 * s2m / L**3 * sq3 + 6 * a2121 * s3m / L**4 * sq5)
        * (2 * a1111 * s3p / L**4 * sq6 + 8 * a1121 * s4p / L**5 * sq2 / 3)
    )
    az = 4 * PI * e2 / dz * (
        a1010 * (1.0 + 2 * a2010 * s3m / L**4 * sq1 + 4 * a2020 * s4m / L**5 * sq2)
        - a2010 * (2 * a1010 * s3m / L**4 * sq1 + 4 * a1020 * s4m / L**5 * sq2)
    )
    az10 = 2 * PI * e2 * np.sqrt(5.0 / 3.0) / dz * (
        a2010 * (1.0 - 4 * a1010 * s2p / L**3 * sq3 - 6 * a1020 * s3p / L**4 * sq4)
        + a1010 * (4 * a2010 * s2p / L**3 * sq3 + 6 * a2020 * s3p / L**4 * sq4)
    )
    ap = 4 * PI * e2 / dp * (
        a1111 * (1.0 - 2 * a2111 * s3p / L**4 * sq6 - 8 * a2121 * s4p / L**5 * sq2 / 3)
        + a2111 * (2 * a1111 * s3p / L**4 * sq6 + 8 * a1121 * s4p / L**5 * sq2 / 3)
    )
    ap10 = 4 * PI * e2 * np.sqrt(5.0) / 3 / dp * (
        a2111 * (1.0 + 2 * a1111 * s2m / L**3 * sq3 + 6 * a1121 * s3m / L**4 * sq5)
        - a1111 * (2 * a2111 * s2m / L**3 * sq3 + 6 * a2121 * s3m / L**4 * sq5)
    )
    return (
        density * ap,
        density * az / e2**2,
        -density * (ap10 + d * ap) * (e1 / e2),
        -density * (az10 + ap10 + d * az + d * ap) / e2,
    )
