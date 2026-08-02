"""Step 5: modified Fresnel reflectivity / transmittance (optics_mod Fresnel_calc)."""

from __future__ import annotations

from granfilm.common.fresnel import fresnel_from_chi
from granfilm.sphere_island.step0_init import InitState
from granfilm.sphere_island.step4_interaction import SurfaceConstitutive


def _scale_r(state: InitState) -> float:
    geom = state.case.geometry.strip().lower()
    return 1.0 if geom in {"film", "2film"} else state.case.R


def step5_fresnel(
    chi: SurfaceConstitutive,
    state: InitState,
    ienergy: int,
) -> float:
    """Return DR/R/DT/T per state.case.out and state.case.fresnel."""
    return fresnel_from_chi(
        chi,
        eps_vacuum=state.eps_vacuum,
        eps_substrate=state.eps_substrate[ienergy],
        energy_ev=float(state.energy[ienergy]),
        R=_scale_r(state),
        theta0=state.case.theta0,
        polarization=state.case.polarization,
        out=state.case.out,
        fresnel_mode=state.case.fresnel,
    )
