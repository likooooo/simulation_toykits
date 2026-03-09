"""Beam field computation: state-free wrappers around simulation.so solvers."""

import numpy as np
from typing import Tuple, List, Dict, Any

from core import simulation_loader

_DEG2RAD = np.pi / 180.0


def _get_sim():
    simulation_loader.get_simulation_module()
    import simulation
    return simulation


def _meta(start_xy: List[float], end_xy: List[float], shape_xy: List[int], wavelength_um: float) -> Dict[str, Any]:
    nx, ny = int(shape_xy[0]), int(shape_xy[1])
    dx = (end_xy[0] - start_xy[0]) / nx if nx else 0.0
    dy = (end_xy[1] - start_xy[1]) / ny if ny else 0.0
    return {
        "nx": nx,
        "ny": ny,
        "dx": dx,
        "dy": dy,
        "wavelength": wavelength_um,
    }


def compute_plane_wave(
    wavelength_um: float,
    theta_deg: float,
    phi_deg: float,
    start_xy: List[float],
    end_xy: List[float],
    shape_xy: List[int],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Plane wave field. Returns (field_ndarray shape (ny, nx), meta)."""
    sim = _get_sim()
    step = [(e - s) / n for s, e, n in zip(start_xy, end_xy, shape_xy)]
    real_start = [a - 0.5 * s for a, s in zip(start_xy, step)]
    theta_rad = theta_deg * _DEG2RAD
    phi_rad = -phi_deg * _DEG2RAD
    gen = sim.plane_wave_solver.generate(
        wavelength_um, theta_rad, phi_rad, real_start, end_xy, shape_xy
    )
    arr = np.array(list(gen)).reshape(shape_xy[1], shape_xy[0])
    return arr, _meta(start_xy, end_xy, shape_xy, wavelength_um)


def compute_quadratic_wave(
    wavelength_um: float,
    z_ratio: float,
    start_xy: List[float],
    end_xy: List[float],
    shape_xy: List[int],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Quadratic wave field. z = z_ratio * wavelength. Returns (field, meta)."""
    sim = _get_sim()
    z = z_ratio * wavelength_um
    gen = sim.quadratic_wave.generate(wavelength_um, z, start_xy, end_xy, shape_xy)
    arr = np.array(list(gen)).reshape(shape_xy[1], shape_xy[0])
    return arr, _meta(start_xy, end_xy, shape_xy, wavelength_um)


def compute_spherical_wave(
    wavelength_um: float,
    z_ratio: float,
    start_xy: List[float],
    end_xy: List[float],
    shape_xy: List[int],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Spherical wave field. z = z_ratio * wavelength. Returns (field, meta)."""
    sim = _get_sim()
    z = z_ratio * wavelength_um
    gen = sim.spherical_wave.generate(wavelength_um, z, start_xy, end_xy, shape_xy)
    arr = np.array(list(gen)).reshape(shape_xy[1], shape_xy[0])
    return arr, _meta(start_xy, end_xy, shape_xy, wavelength_um)


def compute_flat_top_rectangular(
    rx: float,
    ry: float,
    fraction: float,
    order_x: float,
    order_y: float,
    start_xy: List[float],
    end_xy: List[float],
    shape_xy: List[int],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Flat-top beam (rectangular). Returns (field, meta)."""
    sim = _get_sim()
    r = [rx, ry]
    order = [order_x, order_y]
    gen = sim.supper_gaussian_wave.generate_rectangular(
        r, fraction, order, start_xy, end_xy, shape_xy
    )
    arr = np.array(list(gen)).reshape(shape_xy[1], shape_xy[0])
    return arr, _meta(start_xy, end_xy, shape_xy, 0.0)


def compute_flat_top_circular(
    r: float,
    fraction: float,
    order: float,
    start_xy: List[float],
    end_xy: List[float],
    shape_xy: List[int],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Flat-top beam (circular). Returns (field, meta)."""
    sim = _get_sim()
    gen = sim.supper_gaussian_wave.generate_circular(
        r, fraction, order, start_xy, end_xy, shape_xy
    )
    arr = np.array(list(gen)).reshape(shape_xy[1], shape_xy[0])
    return arr, _meta(start_xy, end_xy, shape_xy, 0.0)


def compute_hermite_gaussian(
    m: int,
    n: int,
    wavelength_um: float,
    z: float,
    wx0: float,
    wy0: float,
    start_xy: List[float],
    end_xy: List[float],
    shape_xy: List[int],
    normalize: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Hermite-Gaussian beam. Returns (field, meta). Optionally normalized by max |U|."""
    sim = _get_sim()
    gen = sim.hermite_gaussian_wave.generate(
        m, n, wavelength_um, z, [wx0, wy0], start_xy, end_xy, shape_xy
    )
    arr = np.array(list(gen)).reshape(shape_xy[1], shape_xy[0])
    if normalize:
        peak = np.max(np.abs(arr))
        if peak > 0:
            arr = arr / peak
    return arr, _meta(start_xy, end_xy, shape_xy, wavelength_um)


def compute_laguerre_gaussian(
    p: int,
    l: int,
    wavelength_um: float,
    z: float,
    w0: float,
    start_xy: List[float],
    end_xy: List[float],
    shape_xy: List[int],
    normalize: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Laguerre-Gaussian beam. Returns (field, meta). Optionally normalized by max |U|."""
    sim = _get_sim()
    gen = sim.laguerre_gaussian_wave.generate(
        p, l, wavelength_um, z, w0, start_xy, end_xy, shape_xy
    )
    arr = np.array(list(gen)).reshape(shape_xy[1], shape_xy[0])
    if normalize:
        peak = np.max(np.abs(arr))
        if peak > 0:
            arr = arr / peak
    return arr, _meta(start_xy, end_xy, shape_xy, wavelength_um)
