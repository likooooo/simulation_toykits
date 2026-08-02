"""Unit tests for GranFilm Fresnel extensions (T/DT, constitutive_all)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from granfilm.common.constants import C_NM_S, EPS_VACUUM, HBAR_EV_S
from granfilm.common.fresnel import (
    absorp_coef_constitutive,
    absorp_coef_constitutive_all,
    absorp_coef_invariants,
    aspnes_ooc,
    diff_ref_coef_aspnes,
    diff_ref_coef_constitutive,
    diff_ref_coef_constitutive_all,
    diff_ref_coef_invariants,
    diff_tran_coef_constitutive,
    diff_tran_coef_constitutive_all,
    diff_tran_coef_invariants,
    fresnel_from_chi,
    invariants_calc,
    omega_over_c,
)
from granfilm.sphere_island.step4_interaction import SurfaceConstitutive
from granfilm.sphere_island.step5_fresnel import step5_fresnel


@dataclass
class _Chi:
    gamma: complex
    beta: complex
    tau: complex = 0.0 + 0.0j
    delta: complex = 0.0 + 0.0j


# Known surface constitutive at one energy (non-trivial tau for constitutive_all).
_KNOWN = _Chi(
    gamma=0.02 + 0.005j,
    beta=0.001 + 0.0002j,
    tau=(-3.0e-5) + 1.0e-5j,
    delta=2.0e-6 + 5.0e-7j,
)
_ENERGY_EV = 2.5
_R_NM = 12.0
_THETA0 = np.pi / 4.0
_EPS_SUB = 2.25 + 0.01j


def _ooc() -> float:
    return _R_NM * _ENERGY_EV / (HBAR_EV_S * C_NM_S)


def _geom() -> tuple[complex, complex, complex, float, float, complex]:
    e1 = EPS_VACUUM
    n1 = np.sqrt(e1)
    c0 = np.cos(_THETA0)
    s0 = np.sin(_THETA0)
    n2 = np.sqrt(_EPS_SUB)
    st = n1 / n2 * s0
    ct = np.sqrt(1.0 - st**2)
    return e1, n1, n2, c0, s0, ct


def _ref_diff_ref_constitutive(pol: str, out: str) -> float:
    chi = _KNOWN
    ooc = _ooc()
    e1, n1, n2, c0, s0, ct = _geom()
    if pol == "p":
        tmp1 = 1j * ooc
        tmp2 = -tmp1**2 / 4.0 * e1 * chi.beta * chi.gamma * s0**2
        tmp3 = chi.gamma * c0 * ct + n1 * n2 * chi.beta * e1 * s0**2
        tmp4 = chi.gamma * c0 * ct - n1 * n2 * chi.beta * e1 * s0**2
        factor = np.array([n2 * c0 + n1 * ct, n2 * c0 - n1 * ct], dtype=np.complex128)
        fresnel = factor[1] / factor[0]
        reflec = (factor[1] - tmp1 * tmp4 - tmp2 * factor[1]) / (
            factor[0] - tmp1 * tmp3 - tmp2 * factor[0]
        )
    else:
        tmp1 = 1j * ooc
        factor = np.array([n1 * c0 + n2 * ct, n1 * c0 - n2 * ct], dtype=np.complex128)
        fresnel = factor[1] / factor[0]
        reflec = (factor[1] + tmp1 * chi.gamma) / (factor[0] - tmp1 * chi.gamma)
    if out == "DR":
        return float(np.abs(reflec / fresnel) ** 2 - 1.0)
    return float(np.abs(reflec) ** 2)


def _ref_diff_tran_constitutive(pol: str, out: str) -> float:
    chi = _KNOWN
    ooc = _ooc()
    e1, n1, n2, c0, s0, ct = _geom()
    tmp1 = 1j * ooc
    tmp2 = -tmp1**2 / 4.0 * e1 * chi.beta * chi.gamma * s0**2
    tmp3 = chi.gamma * c0 * ct + n1 * n2 * chi.beta * e1 * s0**2
    if pol == "p":
        factor = n2 * c0 + n1 * ct
        fresnel = 2.0 * n1 * c0 / factor
        transmi = 2.0 * n1 * c0 * (1.0 + tmp2) / (factor - tmp1 * tmp3 - tmp2 * factor)
    else:
        factor = n1 * c0 + n2 * ct
        fresnel = 2.0 * n1 * c0 / factor
        transmi = 2.0 * n1 * c0 / (factor - tmp1 * chi.gamma)
    if out == "DT":
        return float(np.abs(transmi / fresnel) ** 2 - 1.0)
    return float(np.real(n2 * ct / n1 / c0) * np.abs(transmi) ** 2)


def _ref_diff_ref_constitutive_all(pol: str, out: str) -> float:
    chi = _KNOWN
    ooc = _ooc()
    e1, n1, n2, c0, s0, ct = _geom()
    ooc2_tau = ooc**2 * chi.tau
    if pol == "p":
        tmp1 = 1j * ooc
        tmp2 = -tmp1**2 / 4.0 * e1 * chi.beta * chi.gamma * s0**2
        tmp3 = chi.gamma * c0 * ct + n1 * n2 * chi.beta * e1 * s0**2
        tmp4 = chi.gamma * c0 * ct - n1 * n2 * chi.beta * e1 * s0**2
        factor1 = n2 * c0 * (1.0 - ooc2_tau) + n1 * ct * (1.0 + ooc2_tau)
        factor2 = n2 * c0 * (1.0 - ooc2_tau) - n1 * ct * (1.0 + ooc2_tau)
        fresnel = (n2 * c0 - n1 * ct) / (n2 * c0 + n1 * ct)
        numer_base = n2 * c0 - n1 * ct
        denom_base = n2 * c0 + n1 * ct
        reflec = (factor2 - tmp1 * tmp4 - tmp2 * numer_base) / (
            factor1 - tmp1 * tmp3 - tmp2 * denom_base
        )
    else:
        tmp1 = 1j * ooc
        factor1 = n1 * c0 * (1.0 + ooc2_tau) + n2 * ct * (1.0 - ooc2_tau)
        factor2 = n1 * c0 * (1.0 + ooc2_tau) - n2 * ct * (1.0 - ooc2_tau)
        fresnel = (n1 * c0 - n2 * ct) / (n1 * c0 + n2 * ct)
        reflec = (factor2 + tmp1 * chi.gamma) / (factor1 - tmp1 * chi.gamma)
    if out == "DR":
        return float(np.abs(reflec / fresnel) ** 2 - 1.0)
    return float(np.abs(reflec) ** 2)


def _ref_diff_tran_constitutive_all(pol: str, out: str) -> float:
    chi = _KNOWN
    ooc = _ooc()
    e1, n1, n2, c0, s0, ct = _geom()
    ooc2_tau = ooc**2 * chi.tau
    tmp1 = 1j * ooc
    tmp2 = -tmp1**2 / 4.0 * e1 * chi.beta * chi.gamma * s0**2
    tmp3 = chi.gamma * c0 * ct + n1 * n2 * chi.beta * e1 * s0**2
    if pol == "p":
        factor = n2 * c0 * (1.0 - ooc2_tau) + n1 * ct * (1.0 + ooc2_tau)
        fresnel = 2.0 * n1 * c0 / (n2 * c0 + n1 * ct)
        denom_base = n2 * c0 + n1 * ct
        transmi = 2.0 * n1 * c0 * (1.0 + tmp2) / (factor - tmp1 * tmp3 - tmp2 * denom_base)
    else:
        factor = n1 * c0 * (1.0 + ooc2_tau) + n2 * ct * (1.0 - ooc2_tau)
        fresnel = 2.0 * n1 * c0 / (n1 * c0 + n2 * ct)
        transmi = 2.0 * n1 * c0 / (factor - tmp1 * chi.gamma)
    if out == "DT":
        return float(np.abs(transmi / fresnel) ** 2 - 1.0)
    return float(np.real(n2 * ct / n1 / c0) * np.abs(transmi) ** 2)


def _inv_geom() -> tuple[float, complex, complex, float, float, complex]:
    e1 = float(np.real(EPS_VACUUM))
    n1 = np.sqrt(e1)
    c0 = np.cos(_THETA0)
    s0 = np.sin(_THETA0)
    e2 = _EPS_SUB
    n2 = np.sqrt(e2)
    st = n1 / n2 * s0
    ct = np.sqrt(1.0 - st**2)
    return e1, n1, n2, c0, s0, ct


def _ref_invariants() -> tuple[complex, complex, complex, float]:
    chi = _KNOWN
    e1 = EPS_VACUUM
    e2 = _EPS_SUB
    i_e = chi.gamma - e2 * e1 * chi.beta
    i_delta_e = chi.delta - 0.5 * (e2 + e1) / (e2 - e1) * chi.gamma * chi.beta
    i_tau = chi.tau - 0.5 * chi.gamma**2 / (e2 - e1)
    i_c = float(np.imag(chi.gamma / (e2 - e1)))
    return i_e, i_delta_e, i_tau, i_c


def _ref_diff_ref_invariants(pol: str, out: str) -> float:
    ooc = _ooc()
    i_e, i_delta_e, i_tau, i_c = _ref_invariants()
    _, n1, n2, c0, s0, ct = _inv_geom()
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
    else:
        fresnel = float(np.abs((n1 * c0 - n2 * ct) / (n1 * c0 + n2 * ct)) ** 2)
        prefactor = np.exp(4.0 * ooc * i_c * n1 * c0)
        tmp1 = 2.0 * ooc**2 * np.real(i_tau) * ((n1 * c0) ** 2 - np.abs(n2 * ct) ** 2)
        tmp2 = 4.0 * ooc**2 * np.imag(i_tau) * n1 * c0 * np.imag(n2 * ct)
        denominator = np.abs(n1 * c0 + n2 * ct) ** 2 + tmp1 + tmp2
        nominator = np.abs(n1 * c0 - n2 * ct) ** 2 + tmp1 - tmp2
        r_val = float(np.real(prefactor * nominator / denominator))
    if out == "DR":
        return r_val / fresnel - 1.0
    return r_val


def _ref_diff_tran_invariants(pol: str, out: str) -> float:
    ooc = _ooc()
    i_e, i_delta_e, i_tau, i_c = _ref_invariants()
    e1, n1, n2, c0, s0, ct = _inv_geom()
    e2 = _EPS_SUB
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
    else:
        fresnel = float(np.real(4.0 * n1 * n2 * c0 * ct) / np.abs(n1 * c0 + n2 * ct) ** 2)
        tmp1 = n1 * c0 + n2 * ct
        tmp2 = n1 * c0 - n2 * ct
        prefactor = np.exp(2.0 * ooc * i_c * tmp2)
        nominator = 4.0 * n1 * n2 * c0 * ct
        denominator = tmp1**2 - 2.0 * ooc**2 * np.real(i_tau) * (e2 - e1)
        t_val = float(np.real(prefactor * nominator / denominator))
    if out == "DT":
        return t_val / fresnel - 1.0
    return t_val


def _ref_diff_ref_aspnes(pol: str) -> float:
    chi = _KNOWN
    e1 = EPS_VACUUM
    e2 = _EPS_SUB
    n1 = np.sqrt(e1)
    c0 = np.cos(_THETA0)
    s0 = np.sin(_THETA0)
    ooc = aspnes_ooc(_R_NM, _ENERGY_EV)
    prefactor = 8.0 * np.pi * c0 * n1 * ooc
    if pol == "s":
        return float(np.imag(prefactor * chi.gamma / (e2 - e1)))
    y1 = (e2 - e1 * s0**2) * chi.gamma - e2**2 * e1 * s0**2 * chi.beta
    y2 = (e1 - e2) * (e1 * s0**2 - e2 * c0**2)
    return float(np.imag(prefactor * y1 / y2))


_COMMON_KW = dict(
    gamma=_KNOWN.gamma,
    beta=_KNOWN.beta,
    eps_vacuum=EPS_VACUUM,
    eps_substrate=_EPS_SUB,
    theta0=_THETA0,
    ooc=_ooc(),
)


class TestFresnelConstitutive:
    @pytest.mark.parametrize("pol", ["p", "s"])
    @pytest.mark.parametrize("out", ["DR", "R"])
    def test_reflect_matches_formula(self, pol: str, out: str) -> None:
        got = diff_ref_coef_constitutive(polarization=pol, out=out, **_COMMON_KW)
        ref = _ref_diff_ref_constitutive(pol, out)
        assert got == pytest.approx(ref, rel=0, abs=1e-14)

    @pytest.mark.parametrize("pol", ["p", "s"])
    @pytest.mark.parametrize("out", ["DT", "T"])
    def test_transmit_matches_formula(self, pol: str, out: str) -> None:
        got = diff_tran_coef_constitutive(polarization=pol, out=out, **_COMMON_KW)
        ref = _ref_diff_tran_constitutive(pol, out)
        assert got == pytest.approx(ref, rel=0, abs=1e-14)


class TestFresnelConstitutiveAll:
    _ALL_KW = dict(tau=_KNOWN.tau, **_COMMON_KW)

    @pytest.mark.parametrize("pol", ["p", "s"])
    @pytest.mark.parametrize("out", ["DR", "R"])
    def test_reflect_matches_formula(self, pol: str, out: str) -> None:
        got = diff_ref_coef_constitutive_all(polarization=pol, out=out, **self._ALL_KW)
        ref = _ref_diff_ref_constitutive_all(pol, out)
        assert got == pytest.approx(ref, rel=0, abs=1e-14)

    @pytest.mark.parametrize("pol", ["p", "s"])
    @pytest.mark.parametrize("out", ["DT", "T"])
    def test_transmit_matches_formula(self, pol: str, out: str) -> None:
        got = diff_tran_coef_constitutive_all(polarization=pol, out=out, **self._ALL_KW)
        ref = _ref_diff_tran_constitutive_all(pol, out)
        assert got == pytest.approx(ref, rel=0, abs=1e-14)

    def test_tau_changes_dr_vs_constitutive(self) -> None:
        dr_first = diff_ref_coef_constitutive_all(
            polarization="p", out="DR", **self._ALL_KW
        )
        dr_zero_tau = diff_ref_coef_constitutive_all(
            polarization="p", out="DR", tau=0.0, **_COMMON_KW
        )
        dr_const = diff_ref_coef_constitutive(polarization="p", out="DR", **_COMMON_KW)
        assert dr_first != pytest.approx(dr_zero_tau)
        assert dr_zero_tau == pytest.approx(dr_const, rel=0, abs=1e-14)


class TestFresnelAbsorption:
    @pytest.mark.parametrize("pol", ["p", "s"])
    def test_absorp_constitutive_conservation(self, pol: str) -> None:
        r = diff_ref_coef_constitutive(polarization=pol, out="R", **_COMMON_KW)
        t = diff_tran_coef_constitutive(polarization=pol, out="T", **_COMMON_KW)
        a = absorp_coef_constitutive(polarization=pol, **_COMMON_KW)
        assert a + r + t == pytest.approx(1.0, rel=0, abs=1e-12)

    @pytest.mark.parametrize("pol", ["p", "s"])
    def test_absorp_constitutive_all_conservation(self, pol: str) -> None:
        kw = dict(tau=_KNOWN.tau, **_COMMON_KW)
        r = diff_ref_coef_constitutive_all(polarization=pol, out="R", **kw)
        t = diff_tran_coef_constitutive_all(polarization=pol, out="T", **kw)
        a = absorp_coef_constitutive_all(polarization=pol, **kw)
        assert a + r + t == pytest.approx(1.0, rel=0, abs=1e-12)

    @pytest.mark.parametrize(
        ("mode", "absorp_fn"),
        [
            ("constitutive", absorp_coef_constitutive),
            ("constitutive_all", absorp_coef_constitutive_all),
        ],
    )
    def test_fresnel_from_chi_absorption(self, mode: str, absorp_fn) -> None:
        pol = "p"
        kw = dict(tau=_KNOWN.tau, **_COMMON_KW) if mode == "constitutive_all" else _COMMON_KW
        got = fresnel_from_chi(
            _KNOWN,
            eps_vacuum=EPS_VACUUM,
            eps_substrate=_EPS_SUB,
            energy_ev=_ENERGY_EV,
            R=_R_NM,
            theta0=_THETA0,
            polarization=pol,
            out="A",
            fresnel_mode=mode,
        )
        ref = absorp_fn(polarization=pol, **kw)
        assert got == pytest.approx(ref, rel=0, abs=1e-14)


class TestFresnelInvariants:
    def test_invariants_calc(self) -> None:
        inv = invariants_calc(
            gamma=_KNOWN.gamma,
            beta=_KNOWN.beta,
            tau=_KNOWN.tau,
            delta=_KNOWN.delta,
            eps_vacuum=EPS_VACUUM,
            eps_substrate=_EPS_SUB,
        )
        i_e, i_delta_e, i_tau, i_c = _ref_invariants()
        assert inv.e == pytest.approx(i_e, rel=0, abs=1e-20)
        assert inv.delta_e == pytest.approx(i_delta_e, rel=0, abs=1e-20)
        assert inv.tau == pytest.approx(i_tau, rel=0, abs=1e-20)
        assert inv.c == pytest.approx(i_c, rel=0, abs=1e-20)

    @pytest.mark.parametrize("pol", ["p", "s"])
    @pytest.mark.parametrize("out", ["DR", "R"])
    def test_reflect_matches_formula(self, pol: str, out: str) -> None:
        inv = invariants_calc(
            gamma=_KNOWN.gamma,
            beta=_KNOWN.beta,
            tau=_KNOWN.tau,
            delta=_KNOWN.delta,
            eps_vacuum=EPS_VACUUM,
            eps_substrate=_EPS_SUB,
        )
        got = diff_ref_coef_invariants(
            inv=inv,
            eps_vacuum=EPS_VACUUM,
            eps_substrate=_EPS_SUB,
            theta0=_THETA0,
            ooc=_ooc(),
            polarization=pol,
            out=out,
        )
        assert got == pytest.approx(_ref_diff_ref_invariants(pol, out), rel=0, abs=1e-14)

    @pytest.mark.parametrize("pol", ["p", "s"])
    @pytest.mark.parametrize("out", ["DT", "T"])
    def test_transmit_matches_formula(self, pol: str, out: str) -> None:
        inv = invariants_calc(
            gamma=_KNOWN.gamma,
            beta=_KNOWN.beta,
            tau=_KNOWN.tau,
            delta=_KNOWN.delta,
            eps_vacuum=EPS_VACUUM,
            eps_substrate=_EPS_SUB,
        )
        got = diff_tran_coef_invariants(
            inv=inv,
            eps_vacuum=EPS_VACUUM,
            eps_substrate=_EPS_SUB,
            theta0=_THETA0,
            ooc=_ooc(),
            polarization=pol,
            out=out,
        )
        assert got == pytest.approx(_ref_diff_tran_invariants(pol, out), rel=0, abs=1e-14)


class TestFresnelAspnes:
    @pytest.mark.parametrize("pol", ["p", "s"])
    def test_aspnes_matches_formula(self, pol: str) -> None:
        got = diff_ref_coef_aspnes(
            gamma=_KNOWN.gamma,
            beta=_KNOWN.beta,
            eps_vacuum=EPS_VACUUM,
            eps_substrate=_EPS_SUB,
            theta0=_THETA0,
            ooc=aspnes_ooc(_R_NM, _ENERGY_EV),
            polarization=pol,
        )
        assert got == pytest.approx(_ref_diff_ref_aspnes(pol), rel=0, abs=1e-14)

    def test_aspnes_ooc_differs_from_omega_over_c(self) -> None:
        assert aspnes_ooc(_R_NM, _ENERGY_EV) == pytest.approx(
            _ooc() / (2.0 * np.pi), rel=0, abs=1e-15
        )


class TestFresnelDispatch:
    @pytest.mark.parametrize(
        ("mode", "out", "pol", "ref_fn"),
        [
            ("constitutive", "DR", "p", _ref_diff_ref_constitutive),
            ("constitutive", "T", "s", _ref_diff_tran_constitutive),
            ("constitutive_all", "R", "p", _ref_diff_ref_constitutive_all),
            ("constitutive_all", "DT", "s", _ref_diff_tran_constitutive_all),
            ("invariants", "DR", "p", _ref_diff_ref_invariants),
            ("invariants", "T", "s", _ref_diff_tran_invariants),
            ("aspnes", "DR", "p", lambda pol, out: _ref_diff_ref_aspnes(pol)),
        ],
    )
    def test_fresnel_from_chi(self, mode: str, out: str, pol: str, ref_fn) -> None:
        got = fresnel_from_chi(
            _KNOWN,
            eps_vacuum=EPS_VACUUM,
            eps_substrate=_EPS_SUB,
            energy_ev=_ENERGY_EV,
            R=_R_NM,
            theta0=_THETA0,
            polarization=pol,
            out=out,
            fresnel_mode=mode,
        )
        assert got == pytest.approx(ref_fn(pol, out), rel=0, abs=1e-14)

    def test_aspnes_rejects_non_dr(self) -> None:
        with pytest.raises(ValueError, match="not implemented"):
            fresnel_from_chi(
                _KNOWN,
                eps_vacuum=EPS_VACUUM,
                eps_substrate=_EPS_SUB,
                energy_ev=_ENERGY_EV,
                R=_R_NM,
                theta0=_THETA0,
                polarization="p",
                out="R",
                fresnel_mode="aspnes",
            )

    def test_omega_over_c(self) -> None:
        assert omega_over_c(_R_NM, _ENERGY_EV) == pytest.approx(_ooc(), rel=0, abs=1e-20)


class TestStep5Integration:
    def test_step5_t_constitutive_all(self) -> None:
        from granfilm.sphere_island import case as case_mod
        from granfilm.sphere_island.step0_init import InitState

        base = case_mod.default_sphere_case()
        patched = case_mod.GranFilmCase(
            **{
                **base.__dict__,
                "R": _R_NM,
                "out": "T",
                "polarization": "p",
                "fresnel": "constitutive_all",
                "theta0": _THETA0,
                "Nenergy": 1,
                "energy_min": _ENERGY_EV,
                "energy_max": _ENERGY_EV,
            }
        )
        state = InitState(
            case=patched,
            energy=np.array([_ENERGY_EV]),
            eps_island=np.array([1.0 + 0.0j]),
            eps_substrate=np.array([_EPS_SUB]),
            eps_vacuum=EPS_VACUUM,
            density=0.01,
            Rapparent=patched.R,
            coverage=0.1,
            volume=1.0,
            SR=1.0,
            above=True,
            m_max=1,
            theta0_calc=_THETA0,
            phi0_calc=0.0,
            div_surface=1.0,
            lattice_const_eff=1.0,
        )
        chi = SurfaceConstitutive(
            gamma=_KNOWN.gamma,
            beta=_KNOWN.beta,
            tau=_KNOWN.tau,
            delta=_KNOWN.delta,
        )
        got = step5_fresnel(chi, state, 0)
        ref = _ref_diff_tran_constitutive_all("p", "T")
        assert got == pytest.approx(ref, rel=0, abs=1e-14)
