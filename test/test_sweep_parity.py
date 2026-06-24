"""Parity tests: C++ batch sweep vs filmstack_simulation.simulation."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest

_RTOL = 1e-10
_ATOL = 1e-12

ComputeRT = Callable[[Any, float, float], tuple[float, float, float, float]]


def _assert_rt_quad(
    r_s: float, t_s: float, r_p: float, t_p: float, expected: tuple[float, float, float, float]
) -> None:
    assert r_s == pytest.approx(expected[0], rel=_RTOL, abs=_ATOL)
    assert t_s == pytest.approx(expected[1], rel=_RTOL, abs=_ATOL)
    assert r_p == pytest.approx(expected[2], rel=_RTOL, abs=_ATOL)
    assert t_p == pytest.approx(expected[3], rel=_RTOL, abs=_ATOL)


def _assert_angle_sweep_vs_compute_rt(
    layers: Any,
    compute_rt: ComputeRT,
    batch: dict[str, np.ndarray],
    wl_um: float,
    angles_deg: np.ndarray,
) -> None:
    for i, ang in enumerate(angles_deg):
        _assert_rt_quad(
            batch["R_s"][i],
            batch["T_s"][i],
            batch["R_p"][i],
            batch["T_p"][i],
            compute_rt(layers, np.deg2rad(float(ang)), wl_um),
        )


def _assert_wavelength_sweep_vs_compute_rt(
    layers: Any,
    compute_rt: ComputeRT,
    batch: dict[str, np.ndarray],
    angle_deg: float,
    wls_um: np.ndarray,
) -> None:
    th_rad = np.deg2rad(angle_deg)
    for i, wl in enumerate(wls_um):
        _assert_rt_quad(
            batch["R_s"][i],
            batch["T_s"][i],
            batch["R_p"][i],
            batch["T_p"][i],
            compute_rt(layers, th_rad, float(wl)),
        )


def _assert_angular_map_vs_compute_rt(
    layers: Any,
    compute_rt: ComputeRT,
    data: dict[str, np.ndarray],
    wls_um: np.ndarray,
    angles_deg: np.ndarray,
) -> None:
    for j, wl in enumerate(wls_um):
        for i, ang in enumerate(angles_deg):
            _assert_rt_quad(
                data["R_s"][i, j],
                data["T_s"][i, j],
                data["R_p"][i, j],
                data["T_p"][i, j],
                compute_rt(layers, np.deg2rad(float(ang)), float(wl)),
            )


def _power_rt(layers: Any, th_rad: float, wl_um: float) -> tuple[float, float, float, float]:
    from filmstack_simulation.simulation import compute_rt_and_coefficients

    power, _ = compute_rt_and_coefficients(layers, th_rad, wl_um)
    return power


class TestSweepParity:
    def test_angle_sweep_matches_compute_rt(self, sample_tmm_layers):
        from filmstack_simulation.sweep import compute_angle_vs_RT_data

        angles_deg = np.linspace(0, 60, 13)
        batch = compute_angle_vs_RT_data(sample_tmm_layers, 0.532, angles_deg)
        _assert_angle_sweep_vs_compute_rt(
            sample_tmm_layers, _power_rt, batch, 0.532, angles_deg
        )

    def test_wavelength_sweep_matches_compute_rt(self, sample_tmm_layers):
        from filmstack_simulation.sweep import compute_wavelength_vs_RT_data

        wls = np.linspace(0.4, 0.7, 11)
        batch = compute_wavelength_vs_RT_data(sample_tmm_layers, wls, 30.0)
        _assert_wavelength_sweep_vs_compute_rt(
            sample_tmm_layers, _power_rt, batch, 30.0, wls
        )

    def test_angular_map_power_matches_compute_rt(self, sample_tmm_layers):
        from filmstack_simulation.sweep import grid_rt_polarized_and_coeff, reshape_rt_grid

        wls = np.array([0.5, 0.55])
        angles_deg = np.array([0.0, 30.0, 45.0])
        angles_rad = np.deg2rad(angles_deg)
        r_s, t_s, r_p, t_p, _, _, _, _ = grid_rt_polarized_and_coeff(
            sample_tmm_layers, wls, angles_rad
        )
        r_s_2d, t_s_2d, r_p_2d, t_p_2d = reshape_rt_grid(
            r_s, t_s, r_p, t_p, len(wls), len(angles_deg)
        )
        data = {
            "wavelength_um": wls,
            "angles_deg": angles_deg,
            "R_s": r_s_2d,
            "T_s": t_s_2d,
            "R_p": r_p_2d,
            "T_p": t_p_2d,
        }
        _assert_angular_map_vs_compute_rt(
            sample_tmm_layers, _power_rt, data, wls, angles_deg
        )

    def test_grid_coeff_matches_compute_rt_coefficients(self, sample_tmm_layers):
        from filmstack_simulation.simulation import compute_rt_and_coefficients
        from filmstack_simulation.sweep import grid_rt_coeff_polarized

        wls = np.array([0.5, 0.55])
        angles_deg = np.array([0.0, 30.0])
        angles_rad = np.deg2rad(angles_deg)
        r_s, _, r_p, _ = grid_rt_coeff_polarized(sample_tmm_layers, wls, angles_rad)
        idx = 0
        for wl in wls:
            for ang in angles_deg:
                _, (exp_rs, _, exp_rp, _) = compute_rt_and_coefficients(
                    sample_tmm_layers, np.deg2rad(float(ang)), float(wl)
                )
                assert complex(r_s[idx]) == pytest.approx(complex(exp_rs), rel=_RTOL, abs=_ATOL)
                assert complex(r_p[idx]) == pytest.approx(complex(exp_rp), rel=_RTOL, abs=_ATOL)
                idx += 1

    def test_spectral_map_2d_rt_matches_compute_rt(self, standard_ar_formula):
        from filmstack_simulation.simulation import resolve_stack_with_layers, compute_spectral_map_2d

        _, _, layers = resolve_stack_with_layers(standard_ar_formula, {})
        materials = [lyr.background_material for lyr in layers]
        thicknesses_um = [float(lyr.depth) for lyr in layers]
        cache = compute_spectral_map_2d(
            materials,
            thicknesses_um,
            0.5,
            0.55,
            0.0,
            30.0,
            n_wl=3,
            n_ang=4,
            layers=layers,
        )
        for j, wl in enumerate(cache["wavelength_um"]):
            for i, ang in enumerate(cache["angle_deg"]):
                r_s, t_s, r_p, t_p = _power_rt(
                    layers, np.deg2rad(float(ang)), float(wl)
                )
                r_u = 0.5 * (r_s + r_p)
                t_u = 0.5 * (t_s + t_p)
                assert cache["R"][i, j] == pytest.approx(r_u, rel=_RTOL, abs=_ATOL)
                assert cache["T"][i, j] == pytest.approx(t_u, rel=_RTOL, abs=_ATOL)
