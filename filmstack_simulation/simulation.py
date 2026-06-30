"""TMM primitives, stack parsing, and spectral maps (formula: filmstack_visualizer.layers_from_formula)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import filmstack_visualizer
import simulation
import tmm_utils

DEFAULT_N_WL = 50
DEFAULT_N_ANG = 50


def _cplx(z) -> complex:
    return tmm_utils.cplx_from_py(z)


def nk_at_wavelength(material: Any, wl_um: float) -> complex:
    nk = material.nk_at_wavelength_um(float(wl_um))
    return _cplx(nk)


def combine_polarization_rt(
    r_s: np.ndarray,
    t_s: np.ndarray,
    r_p: np.ndarray,
    t_p: np.ndarray,
    polarization: str,
) -> tuple[np.ndarray, np.ndarray]:
    return filmstack_visualizer._combine_polarization_2d(r_s, t_s, r_p, t_p, polarization)


def compute_rt_and_coefficients(
    layers: List[Any],
    th0_rad: float,
    wl_um: float,
) -> Tuple[Tuple[float, float, float, float], Tuple[complex, complex, complex, complex]]:
    """Single TMM pass: power R/T and complex r/t coefficients."""
    sim, tmm_s, tmm_p, nk0, nks, dir_list = _tmm_at(layers, th0_rad, wl_um)
    R_s, T_s = sim.TMM_get_r_t_power_from_tmm_s(
        tmm_s[-1], nk0, dir_list[0], nks, dir_list[-1]
    )
    R_p, T_p = sim.TMM_get_r_t_power_from_tmm_p(
        tmm_p[-1], nk0, dir_list[0], nks, dir_list[-1]
    )
    r_s, t_s = sim.TMM_get_r_t_from_tmm(tmm_s[-1])
    r_p, t_p = sim.TMM_get_r_t_from_tmm(tmm_p[-1])
    return (
        (float(R_s), float(T_s), float(R_p), float(T_p)),
        (_cplx(r_s), _cplx(t_s), _cplx(r_p), _cplx(t_p)),
    )


def _tmm_at(
    layers: List[Any],
    th0_rad: float,
    wl_um: float,
) -> Tuple[Any, Any, Any, complex, complex, Any]:
    sim = simulation
    wl_um = float(wl_um)
    dir_list = sim.TMM_propagate_direction_s(layers, th0_rad, wl_um)
    tmm_s = sim.TMM_interface_transfer_matrix_with_thickness_s(layers, dir_list, wl_um)
    tmm_p = sim.TMM_interface_transfer_matrix_with_thickness_p(layers, dir_list, wl_um)
    nk0 = _cplx(layers[0].background_material.nk_at_wavelength_um(wl_um))
    nks = _cplx(layers[-1].background_material.nk_at_wavelength_um(wl_um))
    return sim, tmm_s, tmm_p, nk0, nks, dir_list


def build_tmm_layers(
    materials: Sequence[Any],
    thicknesses_um: Sequence[float],
) -> List[Any]:
    return filmstack_visualizer.build_tmm_layers(
        materials,
        thicknesses_um,
        simulation_module=simulation,
    )


def resolve_stack_with_layers(
    formula: str,
    materials_db: Dict[str, Any] | None,
) -> Tuple[List[Any], List[float], List[Any]]:
    """Parse formula and return (materials, thicknesses_um, layers)."""
    materials, thicknesses_um, layers = filmstack_visualizer.resolve_formula_stack(
        formula, materials_db or {}, simulation_module=simulation
    )
    return list(materials), list(thicknesses_um), list(layers)


def resolve_stack(
    formula: str,
    materials_db: Dict[str, Any] | None,
) -> Tuple[List[Any], List[float]]:
    """Parse formula and return (materials, thicknesses_um)."""
    materials, thicknesses_um, _ = resolve_stack_with_layers(formula, materials_db)
    return materials, thicknesses_um


def _layers_or_build(
    materials: Sequence[Any],
    thicknesses_um: Sequence[float],
    layers: Sequence[Any] | None,
) -> list[Any]:
    if layers is not None:
        return list(layers)
    return build_tmm_layers(materials, thicknesses_um)


def _psi_delta_from_coeff_maps(
    r_s_map: np.ndarray,
    r_p_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return filmstack_visualizer.compute_psi_delta(r_s_map, r_p_map)


def compute_spectral_map_2d(
    materials: Sequence[Any],
    thicknesses_um: Sequence[float],
    wl_from: float,
    wl_to: float,
    ang_from: float,
    ang_to: float,
    *,
    n_wl: int = DEFAULT_N_WL,
    n_ang: int = DEFAULT_N_ANG,
    layers: Sequence[Any] | None = None,
    polarization: str = "UNPOLARIZED",
) -> Dict[str, Any]:
    """R/T (per polarization) and Psi/Delta on a wavelength-angle grid."""
    from filmstack_simulation.sweep import grid_rt_polarized_power_maps

    wls = np.linspace(float(wl_from), float(wl_to), int(n_wl))
    angles_deg = np.linspace(float(ang_from), float(ang_to), int(n_ang))
    layer_list = _layers_or_build(materials, thicknesses_um, layers)
    angles_rad = np.deg2rad(angles_deg)
    incoherent = filmstack_visualizer.layers_has_incoherent(layer_list)
    r_s_2d, t_s_2d, r_p_2d, t_p_2d, coeff_r_s_2d, coeff_r_p_2d = grid_rt_polarized_power_maps(
        layer_list, wls, angles_rad
    )
    r_map, t_map = combine_polarization_rt(
        r_s_2d, t_s_2d, r_p_2d, t_p_2d, polarization
    )
    if incoherent:
        psi_map, delta_map = None, None
    else:
        psi_map, delta_map = _psi_delta_from_coeff_maps(coeff_r_s_2d, coeff_r_p_2d)
    return {
        "wavelength_um": wls,
        "angle_deg": angles_deg,
        "R": r_map,
        "T": t_map,
        "Psi": psi_map,
        "Delta": delta_map,
        "materials": list(materials),
        "thicknesses_um": list(thicknesses_um),
    }


def _slice_with_psi_delta(
    x: np.ndarray,
    r_arr: np.ndarray,
    t_arr: np.ndarray,
    r_s: np.ndarray,
    r_p: np.ndarray,
    *,
    incoherent: bool = False,
) -> Dict[str, Any]:
    if incoherent:
        return {"x": x, "R": r_arr, "T": t_arr, "Psi": None, "Delta": None}
    psi_arr, delta_arr = _psi_delta_from_coeff_maps(r_s, r_p)
    return {"x": x, "R": r_arr, "T": t_arr, "Psi": psi_arr, "Delta": delta_arr}


def _compute_polarized_curve(
    materials: Sequence[Any],
    thicknesses_um: Sequence[float],
    x: np.ndarray,
    sweep_wls: np.ndarray,
    sweep_angles: np.ndarray,
    *,
    layers: Sequence[Any] | None = None,
    polarization: str = "UNPOLARIZED",
) -> Dict[str, Any]:
    from filmstack_simulation.sweep import grid_rt_polarized_and_coeff

    layer_list = _layers_or_build(materials, thicknesses_um, layers)
    incoherent = filmstack_visualizer.layers_has_incoherent(layer_list)
    r_s, t_s, r_p, t_p, cr_s, _, cr_p, _ = grid_rt_polarized_and_coeff(
        layer_list, sweep_wls, sweep_angles
    )
    r_arr, t_arr = combine_polarization_rt(r_s, t_s, r_p, t_p, polarization)
    return _slice_with_psi_delta(x, r_arr, t_arr, cr_s, cr_p, incoherent=incoherent)


def compute_polarized_curve_at_wavelength(
    materials: Sequence[Any],
    thicknesses_um: Sequence[float],
    target_wl: float,
    ang_from: float,
    ang_to: float,
    *,
    n_ang: int = DEFAULT_N_ANG,
    layers: Sequence[Any] | None = None,
    polarization: str = "UNPOLARIZED",
) -> Dict[str, Any]:
    """R/T and Psi/Delta vs angle at a fixed wavelength."""
    angles_deg = np.linspace(float(ang_from), float(ang_to), int(n_ang))
    return _compute_polarized_curve(
        materials,
        thicknesses_um,
        angles_deg,
        np.asarray([float(target_wl)]),
        np.deg2rad(angles_deg),
        layers=layers,
        polarization=polarization,
    )


def compute_polarized_curve_at_angle(
    materials: Sequence[Any],
    thicknesses_um: Sequence[float],
    target_ang: float,
    wl_from: float,
    wl_to: float,
    *,
    n_wl: int = DEFAULT_N_WL,
    layers: Sequence[Any] | None = None,
    polarization: str = "UNPOLARIZED",
) -> Dict[str, Any]:
    """R/T and Psi/Delta vs wavelength at a fixed angle."""
    wls = np.linspace(float(wl_from), float(wl_to), int(n_wl))
    return _compute_polarized_curve(
        materials,
        thicknesses_um,
        wls,
        wls,
        np.asarray([np.deg2rad(float(target_ang))]),
        layers=layers,
        polarization=polarization,
    )


def compute_rta_at_angle(
    materials: Sequence[Any],
    thicknesses_um: Sequence[float],
    angle_deg: float,
    wl_from: float,
    wl_to: float,
    *,
    n_wl: int = 80,
    layers: Sequence[Any] | None = None,
    polarization: str = "UNPOLARIZED",
) -> Dict[str, np.ndarray]:
    """R/T and A=1-R-T vs wavelength at fixed angle for the given polarization."""
    from filmstack_simulation.sweep import spectrum_rt_polarized

    wls = np.linspace(float(wl_from), float(wl_to), int(n_wl))
    layer_list = _layers_or_build(materials, thicknesses_um, layers)
    angle_rad = float(np.deg2rad(angle_deg))
    r_s, t_s, r_p, t_p = spectrum_rt_polarized(layer_list, wls, angle_rad)
    r_arr, t_arr = combine_polarization_rt(r_s, t_s, r_p, t_p, polarization)
    a_arr = 1.0 - r_arr - t_arr
    return {"wl": wls, "R": r_arr, "T": t_arr, "A": a_arr}
