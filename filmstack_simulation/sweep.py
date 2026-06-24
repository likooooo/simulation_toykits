"""C++-backed TMM wavelength/angle sweeps (μm throughout)."""

from __future__ import annotations

from typing import Any

import numpy as np

import simulation
import tmm_utils


def _cplx_array(values) -> np.ndarray:
    """Convert C++ complex vector output to a numpy complex array."""
    flat = np.ravel(np.asarray(values, dtype=object))
    if flat.size == 0:
        return np.asarray([], dtype=complex)
    sample = flat[0]
    if isinstance(sample, complex):
        return np.asarray(values, dtype=complex)
    if callable(getattr(sample, "real", None)):
        return np.array([tmm_utils.cplx_from_py(z) for z in flat], dtype=complex)
    return np.asarray(values, dtype=complex)


def _as_float_list(arr) -> list[float]:
    return list(np.atleast_1d(np.asarray(arr, dtype=float)))


def _as_power_rtuple(result: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r_s, t_s, r_p, t_p = result
    return (
        np.asarray(r_s, dtype=float),
        np.asarray(t_s, dtype=float),
        np.asarray(r_p, dtype=float),
        np.asarray(t_p, dtype=float),
    )


def _invoke_grid_polarized(
    layers: list[Any],
    wls_um: np.ndarray,
    angles_rad: np.ndarray,
    solver_name: str,
) -> tuple:
    sim = simulation
    return getattr(sim, solver_name)(layers, _as_float_list(wls_um), _as_float_list(angles_rad))


def spectrum_rt_polarized(
    layers: list[Any],
    wls_um: np.ndarray,
    angle_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parallel R/T for s and p vs wavelength (μm)."""
    sim = simulation
    result = sim.TMM_solver_spectrum_rt_polarized_s(
        layers, _as_float_list(wls_um), float(angle_rad)
    )
    return _as_power_rtuple(result)


def angle_rt_polarized(
    layers: list[Any],
    wl_um: float,
    angles_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parallel R/T for s and p vs angle at a fixed wavelength (μm)."""
    result = _invoke_grid_polarized(
        layers, np.asarray([float(wl_um)]), angles_rad, "TMM_solver_grid_rt_polarized_s"
    )
    return _as_power_rtuple(result)


def grid_rt_polarized_and_coeff(
    layers: list[Any],
    wls_um: np.ndarray,
    angles_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flattened (n_wl * n_angle) power R/T and complex r/t coefficients."""
    r_s, t_s, r_p, t_p, cr_s, ct_s, cr_p, ct_p = _invoke_grid_polarized(
        layers, wls_um, angles_rad, "TMM_solver_grid_rt_polarized_and_coeff_s"
    )
    return (
        np.asarray(r_s, dtype=float),
        np.asarray(t_s, dtype=float),
        np.asarray(r_p, dtype=float),
        np.asarray(t_p, dtype=float),
        _cplx_array(cr_s),
        _cplx_array(ct_s),
        _cplx_array(cr_p),
        _cplx_array(ct_p),
    )


def grid_rt_coeff_polarized(
    layers: list[Any],
    wls_um: np.ndarray,
    angles_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flattened (n_wl * n_angle) complex r/t coefficients."""
    r_s, t_s, r_p, t_p = _invoke_grid_polarized(
        layers, wls_um, angles_rad, "TMM_solver_grid_rt_coeff_polarized_s"
    )
    return _cplx_array(r_s), _cplx_array(t_s), _cplx_array(r_p), _cplx_array(t_p)


def reshape_rt_grid(
    r_s: np.ndarray,
    t_s: np.ndarray,
    r_p: np.ndarray,
    t_p: np.ndarray,
    n_wl: int,
    n_ang: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reshape flattened (n_wl * n_ang) sweeps to (n_ang, n_wl) maps."""
    return (
        r_s.reshape(n_wl, n_ang).T,
        t_s.reshape(n_wl, n_ang).T,
        r_p.reshape(n_wl, n_ang).T,
        t_p.reshape(n_wl, n_ang).T,
    )


def grid_rt_polarized_power_maps(
    layers: list[Any],
    wls_um: np.ndarray,
    angles_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """2D (n_ang, n_wl) power R/T maps and complex r coefficients for Psi/Delta."""
    r_s, t_s, r_p, t_p, cr_s, _, cr_p, _ = grid_rt_polarized_and_coeff(layers, wls_um, angles_rad)
    n_wl, n_ang = len(wls_um), len(angles_rad)
    r_s_2d, t_s_2d, r_p_2d, t_p_2d = reshape_rt_grid(r_s, t_s, r_p, t_p, n_wl, n_ang)
    cr_s_2d, _, cr_p_2d, _ = reshape_rt_grid(cr_s, cr_s, cr_p, cr_p, n_wl, n_ang)
    return r_s_2d, t_s_2d, r_p_2d, t_p_2d, cr_s_2d, cr_p_2d


def compute_wavelength_vs_RT_data(
    layers: list[Any],
    wls_um: np.ndarray,
    angle_deg: float,
) -> dict[str, Any]:
    angle_rad = np.deg2rad(angle_deg)
    r_s, t_s, r_p, t_p = spectrum_rt_polarized(layers, wls_um, angle_rad)
    return {
        "wavelength_um": np.asarray(wls_um, dtype=float),
        "R_s": r_s,
        "R_p": r_p,
        "T_s": t_s,
        "T_p": t_p,
        "angle_deg": float(angle_deg),
    }


def compute_angle_vs_RT_data(
    layers: list[Any],
    wl_um: float,
    angles_deg: np.ndarray,
) -> dict[str, Any]:
    angles_deg_arr = np.atleast_1d(np.asarray(angles_deg, dtype=float))
    angles_rad = np.deg2rad(angles_deg_arr)
    r_s, t_s, r_p, t_p = angle_rt_polarized(layers, float(wl_um), angles_rad)
    return {
        "angles_deg": angles_deg_arr,
        "R_s": r_s,
        "R_p": r_p,
        "T_s": t_s,
        "T_p": t_p,
        "wavelength_um": float(wl_um),
    }
