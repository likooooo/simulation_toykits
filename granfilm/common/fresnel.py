"""Fresnel coefficients from surface constitutive chi (GranFilm optics_mod)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from granfilm.common.constants import C_NM_S, HBAR_EV_S


class ChiLike(Protocol):
    gamma: complex
    beta: complex
    tau: complex
    delta: complex


@dataclass(frozen=True)
class Invariants:
    """Dimensionless invariants (Bedeaux book Eq. 3.9.4)."""

    e: complex
    delta_e: complex
    tau: complex
    c: float


def omega_over_c(R: float, energy_ev: float) -> float:
    """Dimensionless omega/c = R * energy / (hbar * c) (optics_mod)."""
    return R * energy_ev / (HBAR_EV_S * C_NM_S)


def aspnes_ooc(R: float, energy_ev: float) -> float:
    """Aspnes 1/lambda dimensionless = R * energy / (2*pi * hbar * c)."""
    return R * energy_ev / (2.0 * np.pi * HBAR_EV_S * C_NM_S)


def _geometry(
    eps_vacuum: complex,
    eps_substrate: complex,
    theta0: float,
) -> tuple[complex, complex, complex, float, float, complex]:
    e1 = eps_vacuum
    n1 = np.sqrt(e1)
    c0 = np.cos(theta0)
    s0 = np.sin(theta0)
    n2 = np.sqrt(eps_substrate)
    st = n1 / n2 * s0
    ct = np.sqrt(1.0 - st**2)
    return e1, n1, n2, c0, s0, ct


def diff_ref_coef_constitutive(
    *,
    gamma: complex,
    beta: complex,
    eps_vacuum: complex,
    eps_substrate: complex,
    theta0: float,
    ooc: float,
    polarization: str,
    out: str,
) -> float:
    """diff_ref_coef_constitutive (Haarmans & Bedeaux, Thin Solid Films 224, Eq. 34-37)."""
    e1, n1, n2, c0, s0, ct = _geometry(eps_vacuum, eps_substrate, theta0)
    pol = polarization.strip().lower()
    out_u = out.strip().upper()

    if pol == "p":
        tmp1 = 1j * ooc
        tmp2 = -tmp1**2 / 4.0 * e1 * beta * gamma * s0**2
        tmp3 = gamma * c0 * ct + n1 * n2 * beta * e1 * s0**2
        tmp4 = gamma * c0 * ct - n1 * n2 * beta * e1 * s0**2
        factor = np.array([n2 * c0 + n1 * ct, n2 * c0 - n1 * ct], dtype=np.complex128)
        fresnel = factor[1] / factor[0]
        reflec = (factor[1] - tmp1 * tmp4 - tmp2 * factor[1]) / (
            factor[0] - tmp1 * tmp3 - tmp2 * factor[0]
        )
    elif pol == "s":
        tmp1 = 1j * ooc
        factor = np.array([n1 * c0 + n2 * ct, n1 * c0 - n2 * ct], dtype=np.complex128)
        fresnel = factor[1] / factor[0]
        reflec = (factor[1] + tmp1 * gamma) / (factor[0] - tmp1 * gamma)
    else:
        raise ValueError(f"polarization {pol!r} not supported")

    if out_u == "DR":
        return float(np.abs(reflec / fresnel) ** 2 - 1.0)
    if out_u == "R":
        return float(np.abs(reflec) ** 2)
    raise ValueError(f"output {out_u!r} not supported for diff_ref_coef_constitutive")


def diff_tran_coef_constitutive(
    *,
    gamma: complex,
    beta: complex,
    eps_vacuum: complex,
    eps_substrate: complex,
    theta0: float,
    ooc: float,
    polarization: str,
    out: str,
) -> float:
    """diff_tran_coef_constitutive (Haarmans & Bedeaux / Bedeaux Physica 67)."""
    e1, n1, n2, c0, s0, ct = _geometry(eps_vacuum, eps_substrate, theta0)
    pol = polarization.strip().lower()
    out_u = out.strip().upper()

    tmp1 = 1j * ooc
    tmp2 = -tmp1**2 / 4.0 * e1 * beta * gamma * s0**2
    tmp3 = gamma * c0 * ct + n1 * n2 * beta * e1 * s0**2

    if pol == "p":
        factor = n2 * c0 + n1 * ct
        fresnel = 2.0 * n1 * c0 / factor
        transmi = 2.0 * n1 * c0 * (1.0 + tmp2) / (factor - tmp1 * tmp3 - tmp2 * factor)
    elif pol == "s":
        factor = n1 * c0 + n2 * ct
        fresnel = 2.0 * n1 * c0 / factor
        transmi = 2.0 * n1 * c0 / (factor - tmp1 * gamma)
    else:
        raise ValueError(f"polarization {pol!r} not supported")

    if out_u == "DT":
        return float(np.abs(transmi / fresnel) ** 2 - 1.0)
    if out_u == "T":
        return float(np.real(n2 * ct / n1 / c0) * np.abs(transmi) ** 2)
    raise ValueError(f"output {out_u!r} not supported for diff_tran_coef_constitutive")


def diff_ref_coef_constitutive_all(
    *,
    gamma: complex,
    beta: complex,
    tau: complex,
    eps_vacuum: complex,
    eps_substrate: complex,
    theta0: float,
    ooc: float,
    polarization: str,
    out: str,
) -> float:
    """diff_ref_coef_constitutive_all (Bedeaux book Eq. 4.18, second-order tau terms)."""
    e1, n1, n2, c0, s0, ct = _geometry(eps_vacuum, eps_substrate, theta0)
    pol = polarization.strip().lower()
    out_u = out.strip().upper()
    ooc2_tau = ooc**2 * tau

    if pol == "p":
        tmp1 = 1j * ooc
        tmp2 = -tmp1**2 / 4.0 * e1 * beta * gamma * s0**2
        tmp3 = gamma * c0 * ct + n1 * n2 * beta * e1 * s0**2
        tmp4 = gamma * c0 * ct - n1 * n2 * beta * e1 * s0**2
        factor1 = n2 * c0 * (1.0 - ooc2_tau) + n1 * ct * (1.0 + ooc2_tau)
        factor2 = n2 * c0 * (1.0 - ooc2_tau) - n1 * ct * (1.0 + ooc2_tau)
        fresnel = (n2 * c0 - n1 * ct) / (n2 * c0 + n1 * ct)
        denom_base = n2 * c0 + n1 * ct
        numer_base = n2 * c0 - n1 * ct
        reflec = (factor2 - tmp1 * tmp4 - tmp2 * numer_base) / (
            factor1 - tmp1 * tmp3 - tmp2 * denom_base
        )
    elif pol == "s":
        tmp1 = 1j * ooc
        factor1 = n1 * c0 * (1.0 + ooc2_tau) + n2 * ct * (1.0 - ooc2_tau)
        factor2 = n1 * c0 * (1.0 + ooc2_tau) - n2 * ct * (1.0 - ooc2_tau)
        fresnel = (n1 * c0 - n2 * ct) / (n1 * c0 + n2 * ct)
        reflec = (factor2 + tmp1 * gamma) / (factor1 - tmp1 * gamma)
    else:
        raise ValueError(f"polarization {pol!r} not supported")

    if out_u == "DR":
        return float(np.abs(reflec / fresnel) ** 2 - 1.0)
    if out_u == "R":
        return float(np.abs(reflec) ** 2)
    raise ValueError(f"output {out_u!r} not supported for diff_ref_coef_constitutive_all")


def diff_tran_coef_constitutive_all(
    *,
    gamma: complex,
    beta: complex,
    tau: complex,
    eps_vacuum: complex,
    eps_substrate: complex,
    theta0: float,
    ooc: float,
    polarization: str,
    out: str,
) -> float:
    """diff_tran_coef_constitutive_all (second-order tau terms)."""
    e1, n1, n2, c0, s0, ct = _geometry(eps_vacuum, eps_substrate, theta0)
    pol = polarization.strip().lower()
    out_u = out.strip().upper()
    ooc2_tau = ooc**2 * tau

    tmp1 = 1j * ooc
    tmp2 = -tmp1**2 / 4.0 * e1 * beta * gamma * s0**2
    tmp3 = gamma * c0 * ct + n1 * n2 * beta * e1 * s0**2

    if pol == "p":
        factor = n2 * c0 * (1.0 - ooc2_tau) + n1 * ct * (1.0 + ooc2_tau)
        fresnel = 2.0 * n1 * c0 / (n2 * c0 + n1 * ct)
        denom_base = n2 * c0 + n1 * ct
        transmi = 2.0 * n1 * c0 * (1.0 + tmp2) / (factor - tmp1 * tmp3 - tmp2 * denom_base)
    elif pol == "s":
        factor = n1 * c0 * (1.0 + ooc2_tau) + n2 * ct * (1.0 - ooc2_tau)
        fresnel = 2.0 * n1 * c0 / (n1 * c0 + n2 * ct)
        transmi = 2.0 * n1 * c0 / (factor - tmp1 * gamma)
    else:
        raise ValueError(f"polarization {pol!r} not supported")

    if out_u == "DT":
        return float(np.abs(transmi / fresnel) ** 2 - 1.0)
    if out_u == "T":
        return float(np.real(n2 * ct / n1 / c0) * np.abs(transmi) ** 2)
    raise ValueError(f"output {out_u!r} not supported for diff_tran_coef_constitutive_all")


def absorp_coef_constitutive(
    *,
    gamma: complex,
    beta: complex,
    eps_vacuum: complex,
    eps_substrate: complex,
    theta0: float,
    ooc: float,
    polarization: str,
) -> float:
    """absorp_coef_constitutive: A = 1 - R - T (optics_mod Fresnel_calc out='A')."""
    r = diff_ref_coef_constitutive(
        gamma=gamma,
        beta=beta,
        eps_vacuum=eps_vacuum,
        eps_substrate=eps_substrate,
        theta0=theta0,
        ooc=ooc,
        polarization=polarization,
        out="R",
    )
    t = diff_tran_coef_constitutive(
        gamma=gamma,
        beta=beta,
        eps_vacuum=eps_vacuum,
        eps_substrate=eps_substrate,
        theta0=theta0,
        ooc=ooc,
        polarization=polarization,
        out="T",
    )
    return 1.0 - r - t


def absorp_coef_constitutive_all(
    *,
    gamma: complex,
    beta: complex,
    tau: complex,
    eps_vacuum: complex,
    eps_substrate: complex,
    theta0: float,
    ooc: float,
    polarization: str,
) -> float:
    """absorp_coef_constitutive_all: A = 1 - R - T (second-order tau terms)."""
    r = diff_ref_coef_constitutive_all(
        gamma=gamma,
        beta=beta,
        tau=tau,
        eps_vacuum=eps_vacuum,
        eps_substrate=eps_substrate,
        theta0=theta0,
        ooc=ooc,
        polarization=polarization,
        out="R",
    )
    t = diff_tran_coef_constitutive_all(
        gamma=gamma,
        beta=beta,
        tau=tau,
        eps_vacuum=eps_vacuum,
        eps_substrate=eps_substrate,
        theta0=theta0,
        ooc=ooc,
        polarization=polarization,
        out="T",
    )
    return 1.0 - r - t


def invariants_calc(
    *,
    gamma: complex,
    beta: complex,
    tau: complex,
    delta: complex,
    eps_vacuum: complex,
    eps_substrate: complex,
) -> Invariants:
    """Invariants_calc (Bedeaux book Eq. 3.9.4)."""
    e1 = eps_vacuum
    e2 = eps_substrate
    return Invariants(
        e=gamma - e2 * e1 * beta,
        delta_e=delta - 0.5 * (e2 + e1) / (e2 - e1) * gamma * beta,
        tau=tau - 0.5 * gamma**2 / (e2 - e1),
        c=float(np.imag(gamma / (e2 - e1))),
    )


def _invariants_geometry(
    eps_vacuum: complex,
    eps_substrate: complex,
    theta0: float,
) -> tuple[float, complex, complex, float, float, complex]:
    e1 = float(np.real(eps_vacuum))
    n1 = np.sqrt(e1)
    c0 = np.cos(theta0)
    s0 = np.sin(theta0)
    e2 = eps_substrate
    n2 = np.sqrt(e2)
    st = n1 / n2 * s0
    ct = np.sqrt(1.0 - st**2)
    return e1, n1, n2, c0, s0, ct


def diff_ref_coef_invariants(
    *,
    inv: Invariants,
    eps_vacuum: complex,
    eps_substrate: complex,
    theta0: float,
    ooc: float,
    polarization: str,
    out: str,
) -> float:
    """diff_ref_coef_invariants (Bedeaux book Eq. 4.56, 4.62)."""
    _, n1, n2, c0, s0, ct = _invariants_geometry(eps_vacuum, eps_substrate, theta0)
    pol = polarization.strip().lower()
    out_u = out.strip().upper()
    i_e = inv.e
    i_tau = inv.tau
    i_delta_e = inv.delta_e
    i_c = inv.c

    if pol == "p":
        fresnel = float(np.abs((n2 * c0 - n1 * ct) / (n2 * c0 + n1 * ct)) ** 2)
        tmp1 = (
            2.0
            * ooc**2
            * (np.real(i_tau) - np.real(i_delta_e) * (n1 * s0) ** 2)
            * (np.abs(n2) ** 2 * c0**2 - n1**2 * np.abs(ct) ** 2)
        )
        tmp2 = (
            np.imag(n1 * np.conj(n2) * c0 * ct)
            * 4.0
            * ooc**2
            * (np.imag(i_tau) - np.imag(i_delta_e) * (n1 * s0) ** 2)
        )
        prefactor = np.exp(4.0 * ooc * i_c * n1 * c0)
        nominator = (
            np.abs(n2 * c0 - n1 * ct - 1j * ooc * (n1 / n2) * i_e * s0**2) ** 2
            - tmp1
            + tmp2
        )
        denominator = (
            np.abs(n2 * c0 + n1 * ct + 1j * ooc * (n1 / n2) * i_e * s0**2) ** 2
            - tmp1
            - tmp2
        )
        r_val = float(np.real(prefactor * nominator / denominator))
    elif pol == "s":
        fresnel = float(np.abs((n1 * c0 - n2 * ct) / (n1 * c0 + n2 * ct)) ** 2)
        prefactor = np.exp(4.0 * ooc * i_c * n1 * c0)
        tmp1 = 2.0 * ooc**2 * np.real(i_tau) * ((n1 * c0) ** 2 - np.abs(n2 * ct) ** 2)
        tmp2 = 4.0 * ooc**2 * np.imag(i_tau) * n1 * c0 * np.imag(n2 * ct)
        denominator = np.abs(n1 * c0 + n2 * ct) ** 2 + tmp1 + tmp2
        nominator = np.abs(n1 * c0 - n2 * ct) ** 2 + tmp1 - tmp2
        r_val = float(np.real(prefactor * nominator / denominator))
    else:
        raise ValueError(f"polarization {pol!r} not supported")

    if out_u == "DR":
        return r_val / fresnel - 1.0
    if out_u == "R":
        return r_val
    raise ValueError(f"output {out_u!r} not supported for diff_ref_coef_invariants")


def diff_tran_coef_invariants(
    *,
    inv: Invariants,
    eps_vacuum: complex,
    eps_substrate: complex,
    theta0: float,
    ooc: float,
    polarization: str,
    out: str,
) -> float:
    """diff_tran_coef_invariants (Bedeaux book Eq. 4.60, 4.66)."""
    e1, n1, n2, c0, s0, ct = _invariants_geometry(eps_vacuum, eps_substrate, theta0)
    e2 = eps_substrate
    pol = polarization.strip().lower()
    out_u = out.strip().upper()
    i_e = inv.e
    i_tau = inv.tau
    i_delta_e = inv.delta_e
    i_c = inv.c

    if pol == "p":
        fresnel = float(np.real(4.0 * n1 * n2 * c0 * ct) / np.abs(n2 * c0 + n1 * ct) ** 2)
        prefactor = np.exp(2.0 * ooc * i_c * (n1 * c0 - n2 * ct))
        nominator = 4.0 * n1 * n2 * c0 * ct
        tmp1 = n2 * c0 + n1 * ct
        tmp2 = (
            e1 / e2 * np.abs(i_e) ** 2 * s0**4
            - 2.0 * (np.real(i_tau) - np.real(i_delta_e) * e1 * s0**2) * (e2 * c0**2 - e1 * ct**2)
        )
        denominator = tmp1**2 - 2.0 * ooc * n1 / n2 * np.imag(i_e) * s0**2 * tmp1 + ooc**2 * tmp2
        t_val = float(np.real(prefactor * nominator / denominator))
    elif pol == "s":
        fresnel = float(np.real(4.0 * n1 * n2 * c0 * ct) / np.abs(n1 * c0 + n2 * ct) ** 2)
        tmp1 = n1 * c0 + n2 * ct
        tmp2 = n1 * c0 - n2 * ct
        prefactor = np.exp(2.0 * ooc * i_c * tmp2)
        nominator = 4.0 * n1 * n2 * c0 * ct
        denominator = tmp1**2 - 2.0 * ooc**2 * np.real(i_tau) * (e2 - e1)
        t_val = float(np.real(prefactor * nominator / denominator))
    else:
        raise ValueError(f"polarization {pol!r} not supported")

    if out_u == "DT":
        return t_val / fresnel - 1.0
    if out_u == "T":
        return t_val
    raise ValueError(f"output {out_u!r} not supported for diff_tran_coef_invariants")


def absorp_coef_invariants(
    *,
    inv: Invariants,
    eps_vacuum: complex,
    eps_substrate: complex,
    theta0: float,
    ooc: float,
    polarization: str,
) -> float:
    """Absorp_coef_invariants (Bedeaux book Eq. 4.61, 4.67)."""
    e1, n1, n2, c0, s0, ct = _invariants_geometry(eps_vacuum, eps_substrate, theta0)
    e2 = eps_substrate
    pol = polarization.strip().lower()
    i_e = inv.e
    i_tau = inv.tau
    i_delta_e = inv.delta_e
    i_c = inv.c

    if pol == "p":
        prefactor = np.exp(2.0 * ooc * i_c * (n1 * c0 - n2 * ct))
        nominator = 4.0 * ooc * n1 * c0 * (
            i_c * (e2 - e1) * (s0**2 + ct**2)
            - np.imag(i_e) * s0**2
            + 2.0
            * ooc
            / n2
            * s0**2
            * ct**2
            * (i_c**2 * (e2 - e1) ** 2 - i_c * np.imag(i_e) * (e2 - e1))
        )
        tmp1 = n2 * c0 + n1 * ct
        tmp2 = (
            e1 / e2 * np.abs(i_e) ** 2 * s0**4
            - 2.0 * (np.real(i_tau) - np.real(i_delta_e) * e1 * s0**2) * (e2 * c0**2 - e1 * ct**2)
        )
        denominator = tmp1**2 - 2.0 * ooc * n1 / n2 * np.imag(i_e) * s0**2 * tmp1 + ooc**2 * tmp2
        return float(np.real(prefactor * nominator / denominator))
    if pol == "s":
        tmp1 = n1 * c0 - n2 * ct
        tmp2 = ooc * np.real(i_tau) * (e2 - e1)
        prefactor = np.exp(2.0 * ooc * i_c * tmp1)
        nominator = 4.0 * n1 * c0 * tmp2
        denominator = tmp1**2 - 2.0 * ooc * tmp2
        return float(np.real(prefactor * nominator / denominator))
    raise ValueError(f"polarization {pol!r} not supported")


def diff_ref_coef_aspnes(
    *,
    gamma: complex,
    beta: complex,
    eps_vacuum: complex,
    eps_substrate: complex,
    theta0: float,
    ooc: float,
    polarization: str,
) -> float:
    """diff_ref_coef_aspnes (Borensztein Thin Solid Films 125 (1985) 129, Eq. 6-7)."""
    e1 = eps_vacuum
    n1 = np.sqrt(e1)
    c0 = np.cos(theta0)
    s0 = np.sin(theta0)
    e2 = eps_substrate
    pol = polarization.strip().lower()
    prefactor = 8.0 * np.pi * c0 * n1 * ooc

    if pol == "s":
        return float(np.imag(prefactor * gamma / (e2 - e1)))
    if pol == "p":
        y1 = (e2 - e1 * s0**2) * gamma - e2**2 * e1 * s0**2 * beta
        y2 = (e1 - e2) * (e1 * s0**2 - e2 * c0**2)
        return float(np.imag(prefactor * y1 / y2))
    raise ValueError(f"polarization {pol!r} not supported")


def fresnel_from_chi(
    chi: ChiLike,
    *,
    eps_vacuum: complex,
    eps_substrate: complex,
    energy_ev: float,
    R: float,
    theta0: float,
    polarization: str,
    out: str,
    fresnel_mode: str,
) -> float:
    """Dispatch Fresnel_calc constitutive / constitutive_all / invariants / aspnes branches."""
    mode = fresnel_mode.strip().lower()
    out_u = out.strip().upper()
    ooc = omega_over_c(R, energy_ev)
    common = dict(
        gamma=chi.gamma,
        beta=chi.beta,
        eps_vacuum=eps_vacuum,
        eps_substrate=eps_substrate,
        theta0=theta0,
        ooc=ooc,
        polarization=polarization,
    )

    if mode == "constitutive":
        if out_u in {"DR", "R"}:
            return diff_ref_coef_constitutive(out=out, **common)
        if out_u in {"DT", "T"}:
            return diff_tran_coef_constitutive(out=out, **common)
        if out_u == "A":
            return absorp_coef_constitutive(**common)
    elif mode == "constitutive_all":
        all_kw = dict(tau=chi.tau, **common)
        if out_u in {"DR", "R"}:
            return diff_ref_coef_constitutive_all(out=out, **all_kw)
        if out_u in {"DT", "T"}:
            return diff_tran_coef_constitutive_all(out=out, **all_kw)
        if out_u == "A":
            return absorp_coef_constitutive_all(**all_kw)
    elif mode == "invariants":
        inv = invariants_calc(
            gamma=chi.gamma,
            beta=chi.beta,
            tau=chi.tau,
            delta=chi.delta,
            eps_vacuum=eps_vacuum,
            eps_substrate=eps_substrate,
        )
        inv_kw = dict(
            inv=inv,
            eps_vacuum=eps_vacuum,
            eps_substrate=eps_substrate,
            theta0=theta0,
            ooc=ooc,
            polarization=polarization,
        )
        if out_u in {"DR", "R"}:
            return diff_ref_coef_invariants(out=out, **inv_kw)
        if out_u in {"DT", "T"}:
            return diff_tran_coef_invariants(out=out, **inv_kw)
        if out_u == "A":
            return absorp_coef_invariants(**inv_kw)
    elif mode == "aspnes":
        if out_u == "DR":
            return diff_ref_coef_aspnes(
                gamma=chi.gamma,
                beta=chi.beta,
                eps_vacuum=eps_vacuum,
                eps_substrate=eps_substrate,
                theta0=theta0,
                ooc=aspnes_ooc(R, energy_ev),
                polarization=polarization,
            )
    else:
        raise ValueError(f"fresnel mode {mode!r} not implemented")

    raise ValueError(f"output {out_u!r} not implemented for fresnel mode {mode!r}")
