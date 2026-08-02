"""Simulation TMM thickness adjoint for R/T power."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from stack_spec import StackSpec, build_simulation_layers


class Polarization:
    TE = "TE"
    TM = "TM"
    UNPOLARIZED = "UNPOLARIZED"


def _combine_unpolarized(rs: float, rp: float) -> float:
    return 0.5 * (rs + rp)


def simulation_rt_at(
    layers: list[Any],
    wl_um: float,
    angle_deg: float,
    pol: str,
) -> tuple[float, float]:
    from filmstack_optimization_utils import compute_wavelength_vs_RT_data

    data = compute_wavelength_vs_RT_data(layers, np.asarray([wl_um]), float(angle_deg))
    rs, ts = float(data["R_s"][0]), float(data["T_s"][0])
    rp, tp = float(data["R_p"][0]), float(data["T_p"][0])
    if pol == Polarization.TE:
        return rs, ts
    if pol == Polarization.TM:
        return rp, tp
    return _combine_unpolarized(rs, rp), _combine_unpolarized(ts, tp)


def simulation_thickness_rt_derivatives(
    spec: StackSpec,
    materials_db: dict[str, Any],
    wls_nm: Sequence[float],
    angle_deg: float,
    pol: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (dR/dd_um, dT/dd_um) shape (n_sim_layers, n_wl) via simulation adjoint."""
    from filmstack_optimization_utils import (
        Polarization as SimPol,
        thickness_reflectance_adjoint,
        thickness_transmittance_adjoint,
    )

    layers = build_simulation_layers(materials_db, spec)
    sim_pol = {
        Polarization.TE: SimPol.TE,
        Polarization.TM: SimPol.TM,
        Polarization.UNPOLARIZED: SimPol.UNPOLARIZED,
    }[pol]

    wls_nm = list(map(float, wls_nm))
    nb_layers = len(layers)
    nb_wl = len(wls_nm)
    dR = np.zeros((nb_layers, nb_wl), dtype=float)
    dT = np.zeros((nb_layers, nb_wl), dtype=float)
    th0 = np.deg2rad(float(angle_deg))
    for iw, wl_nm in enumerate(wls_nm):
        wl_um = wl_nm / 1000.0
        gR = thickness_reflectance_adjoint(layers, th0, wl_um, 1.0, sim_pol)
        gT = thickness_transmittance_adjoint(layers, th0, wl_um, 1.0, sim_pol)
        dR[:, iw] = np.asarray(gR, dtype=float)
        dT[:, iw] = np.asarray(gT, dtype=float)
    return dR, dT


def simulation_thickness_rt_derivatives_constant_layer(
    spec: StackSpec,
    materials_db: dict[str, Any],
    layer_index: int,
    layer_nk: complex,
    wl_nm: float,
    angle_deg: float,
    pol: str,
) -> tuple[float, float]:
    from dataclasses import replace

    from simulation_paths import simulation_module

    sim = simulation_module()
    const = sim.material_s.from_nk(complex(layer_nk), f"scan_{layer_index}")
    tokens = list(spec.film_tokens)
    tokens[layer_index] = f"_scan_{layer_index}"
    patched_db = dict(materials_db)
    patched_db[tokens[layer_index]] = const
    patched_spec = replace(spec, film_tokens=tokens)
    dR, dT = simulation_thickness_rt_derivatives(
        patched_spec, patched_db, [wl_nm], angle_deg, pol
    )
    sim_layer = 1 + layer_index
    return float(dR[sim_layer, 0]), float(dT[sim_layer, 0])
