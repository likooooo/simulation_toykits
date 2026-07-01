"""OpenFilters abeles thickness derivatives for R/T power."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from abeles_loader import load_abeles
from stack_spec import StackSpec, sample_stack_nk, simulation_nk_to_abeles

# OpenFilters definitions: S=90 (TE), P=0 (TM), UNPOLARIZED=45
POL_TE = 90.0
POL_TM = 0.0
POL_UNPOLARIZED = 45.0


def _fill_n_holder(abeles: Any, wvls: Any, values: Sequence[complex]) -> Any:
    n_obj = abeles.N(wvls)
    for i, nk in enumerate(values):
        n_obj.N[i] = simulation_nk_to_abeles(complex(nk))
    return n_obj


def openfilters_rt_spectrum(
    spec: StackSpec,
    materials_db: dict[str, Any],
    wls_nm: Sequence[float],
    angle_deg: float,
    pol: float,
    *,
    of_root=None,
) -> tuple[np.ndarray, np.ndarray]:
    abeles = load_abeles(of_root)
    wls_nm = list(map(float, wls_nm))
    nb_wl = len(wls_nm)
    wvls = abeles.wvls(nb_wl)
    for i, w in enumerate(wls_nm):
        wvls.set_wvl(i, float(w))

    layers_nk, medium_nk, substrate_nk = sample_stack_nk(materials_db, spec, wls_nm)
    n_medium = _fill_n_holder(abeles, wvls, medium_nk)
    n_substrate = _fill_n_holder(abeles, wvls, substrate_nk)

    sin2_theta_0 = abeles.sin2(wvls)
    sin2_theta_0.set_sin2_theta_0(n_medium, float(angle_deg))

    nb_layers = spec.openfilters_layer_count()
    ppm = abeles.pre_and_post_matrices(wvls, nb_layers)
    for of_i, sim_i in enumerate(range(nb_layers - 1, -1, -1)):
        n_layer = _fill_n_holder(abeles, wvls, layers_nk[sim_i])
        ppm.set_pre_and_post_matrices(
            of_i, n_layer, float(spec.film_thicknesses_nm[sim_i]), sin2_theta_0
        )
    ppm.multiply_pre_and_post_matrices()
    m_global = ppm.get_global_matrices()

    r_and_t = abeles.r_and_t(wvls)
    r_and_t.calculate_r_and_t(m_global, n_medium, n_substrate, sin2_theta_0)
    r_spec = abeles.R(wvls)
    t_spec = abeles.T(wvls)
    r_spec.calculate_R(r_and_t, pol)
    t_spec.calculate_T(r_and_t, n_medium, n_substrate, sin2_theta_0, pol)
    return np.asarray(r_spec.data, dtype=float), np.asarray(t_spec.data, dtype=float)


def openfilters_thickness_rt_derivatives(
    spec: StackSpec,
    materials_db: dict[str, Any],
    wls_nm: Sequence[float],
    angle_deg: float,
    pol: float,
    *,
    of_root=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (dR/dd_nm, dT/dd_nm) shape (n_film, n_wl)."""
    abeles = load_abeles(of_root)
    wls_nm = list(map(float, wls_nm))
    nb_wl = len(wls_nm)
    wvls = abeles.wvls(nb_wl)
    for i, w in enumerate(wls_nm):
        wvls.set_wvl(i, float(w))

    layers_nk, medium_nk, substrate_nk = sample_stack_nk(materials_db, spec, wls_nm)
    n_medium = _fill_n_holder(abeles, wvls, medium_nk)
    n_substrate = _fill_n_holder(abeles, wvls, substrate_nk)

    sin2_theta_0 = abeles.sin2(wvls)
    sin2_theta_0.set_sin2_theta_0(n_medium, float(angle_deg))

    nb_layers = spec.openfilters_layer_count()
    ppm = abeles.pre_and_post_matrices(wvls, nb_layers)
    layer_n: list[Any] = []
    # OpenFilters stacks layers substrate → medium; simulation is incident → exit.
    for of_i, sim_i in enumerate(range(nb_layers - 1, -1, -1)):
        nk_list = layers_nk[sim_i]
        n_layer = _fill_n_holder(abeles, wvls, nk_list)
        layer_n.append(n_layer)
        ppm.set_pre_and_post_matrices(
            of_i, n_layer, float(spec.film_thicknesses_nm[sim_i]), sin2_theta_0
        )
    ppm.multiply_pre_and_post_matrices()
    m_global = ppm.get_global_matrices()

    r_and_t = abeles.r_and_t(wvls)
    r_and_t.calculate_r_and_t(m_global, n_medium, n_substrate, sin2_theta_0)

    d_mi = abeles.dM(wvls)
    d_m = abeles.dM(wvls)
    psi = abeles.psi_matrices(wvls)
    dr_dt = abeles.dr_and_dt(wvls)
    d_r = abeles.dR(wvls)
    d_t = abeles.dT(wvls)
    psi.calculate_psi_matrices(r_and_t, n_medium, n_substrate, sin2_theta_0)

    dR = np.zeros((nb_layers, nb_wl), dtype=float)
    dT = np.zeros((nb_layers, nb_wl), dtype=float)
    for of_k in range(nb_layers):
        sim_k = nb_layers - 1 - of_k
        d_mi.set_dMi_thickness(
            layer_n[of_k], float(spec.film_thicknesses_nm[sim_k]), sin2_theta_0
        )
        d_m.calculate_dM(d_mi, ppm, of_k)
        dr_dt.calculate_dr_and_dt(d_m, psi)
        d_r.calculate_dR(dr_dt, r_and_t, pol)
        d_t.calculate_dT(dr_dt, r_and_t, n_medium, n_substrate, sin2_theta_0, pol)
        for iw in range(nb_wl):
            dR[sim_k, iw] = float(d_r.data[iw])
            dT[sim_k, iw] = float(d_t.data[iw])

    return dR, dT


def openfilters_thickness_rt_derivatives_constant_layer(
    spec: StackSpec,
    materials_db: dict[str, Any],
    layer_index: int,
    layer_nk: complex,
    wls_nm: Sequence[float],
    angle_deg: float,
    pol: float,
    *,
    of_root=None,
) -> tuple[float, float]:
    """Derivatives w.r.t. one layer with overridden constant nk (single wl)."""
    from dataclasses import replace

    from simulation_paths import simulation_module

    if len(wls_nm) != 1:
        raise ValueError("constant_layer helper supports a single wavelength")
    sim = simulation_module()
    scan_tok = f"_scan_{layer_index}"
    patched_db = dict(materials_db)
    patched_db[scan_tok] = sim.material_s.from_nk(complex(layer_nk), scan_tok)
    tokens = list(spec.film_tokens)
    tokens[layer_index] = scan_tok
    patched_spec = replace(spec, film_tokens=tokens)
    dR, dT = openfilters_thickness_rt_derivatives(
        patched_spec, patched_db, wls_nm, angle_deg, pol, of_root=of_root
    )
    return float(dR[layer_index, 0]), float(dT[layer_index, 0])
