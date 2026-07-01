"""Compare OpenFilters and simulation R/T spectra and thickness derivatives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from openfilters_derivatives import (
    POL_TE,
    POL_TM,
    POL_UNPOLARIZED,
    openfilters_rt_spectrum,
    openfilters_thickness_rt_derivatives,
)
from simulation_derivatives import (
    Polarization,
    simulation_rt_at,
    simulation_thickness_rt_derivatives,
)
from stack_spec import StackSpec, build_simulation_layers


@dataclass
class CompareResult:
    ok_rt: bool
    ok_dR: bool
    ok_dT: bool
    max_rt_err: float
    max_dR_err: float
    max_dT_err: float
    message: str


def _pol_map(pol: str) -> float:
    return {
        Polarization.TE: POL_TE,
        Polarization.TM: POL_TM,
        Polarization.UNPOLARIZED: POL_UNPOLARIZED,
    }[pol]


def compare_rt_gate(
    spec: StackSpec,
    materials_db: dict[str, Any],
    wls_nm: Sequence[float],
    angle_deg: float,
    pol: str,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> tuple[bool, float]:
    of_pol = _pol_map(pol)
    r_of, t_of = openfilters_rt_spectrum(spec, materials_db, wls_nm, angle_deg, of_pol)
    layers = build_simulation_layers(materials_db, spec)
    errs: list[float] = []
    for iw, wl_nm in enumerate(wls_nm):
        r_sim, t_sim = simulation_rt_at(layers, wl_nm / 1000.0, angle_deg, pol)
        errs.append(abs(r_of[iw] - r_sim))
        errs.append(abs(t_of[iw] - t_sim))
    max_err = max(errs) if errs else 0.0
    ok = max_err <= max(atol, rtol)
    return ok, max_err


def compare_thickness_derivatives(
    spec: StackSpec,
    materials_db: dict[str, Any],
    wls_nm: Sequence[float],
    angle_deg: float,
    pol: str,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    bookend_atol: float = 1e-12,
) -> CompareResult:
    ok_rt, rt_err = compare_rt_gate(spec, materials_db, wls_nm, angle_deg, pol)
    if not ok_rt:
        return CompareResult(False, False, False, rt_err, np.inf, np.inf, "forward R/T gate failed")

    of_pol = _pol_map(pol)
    dR_of, dT_of = openfilters_thickness_rt_derivatives(
        spec, materials_db, wls_nm, angle_deg, of_pol
    )
    dR_sim, dT_sim = simulation_thickness_rt_derivatives(
        spec, materials_db, wls_nm, angle_deg, pol
    )

    # OpenFilters: per nm; simulation: per um
    dR_of_um = dR_of * 1000.0
    dT_of_um = dT_of * 1000.0

    film_idx = spec.simulation_film_layer_indices()
    dR_errs: list[float] = []
    dT_errs: list[float] = []
    for j, li in enumerate(film_idx):
        for iw in range(len(wls_nm)):
            ref_r = max(abs(dR_sim[li, iw]), 1.0)
            ref_t = max(abs(dT_sim[li, iw]), 1.0)
            dR_errs.append(abs(dR_sim[li, iw] - dR_of_um[j, iw]) / ref_r)
            dT_errs.append(abs(dT_sim[li, iw] - dT_of_um[j, iw]) / ref_t)

    for li in (0, len(dR_sim) - 1):
        if max(np.max(np.abs(dR_sim[li])), np.max(np.abs(dT_sim[li]))) > bookend_atol:
            return CompareResult(
                ok_rt, False, False, rt_err, np.inf, np.inf, f"bookend layer {li} gradient non-zero"
            )

    max_dR = max(dR_errs) if dR_errs else 0.0
    max_dT = max(dT_errs) if dT_errs else 0.0
    ok_dR = max_dR <= rtol
    ok_dT = max_dT <= rtol
    msg = f"rt_err={rt_err:.3e} dR_rel={max_dR:.3e} dT_rel={max_dT:.3e}"
    return CompareResult(ok_rt, ok_dR, ok_dT, rt_err, max_dR, max_dT, msg)
