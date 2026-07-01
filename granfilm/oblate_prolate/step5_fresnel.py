"""Step 5: modified Fresnel reflectivity / transmittance."""

from __future__ import annotations

from granfilm.common.fresnel import fresnel_from_chi
from granfilm.oblate_prolate.step0_init import SpheroidInitState
from granfilm.oblate_prolate.step4_interaction import SurfaceConstitutive


def step5_fresnel(chi: SurfaceConstitutive, state: SpheroidInitState, ienergy: int) -> float:
    return fresnel_from_chi(
        chi,
        eps_vacuum=state.eps_vacuum,
        eps_substrate=state.eps_substrate[ienergy],
        energy_ev=float(state.energy[ienergy]),
        R=state.R,
        theta0=state.case.theta0,
        polarization=state.case.polarization,
        out=state.case.out,
        fresnel_mode=state.case.fresnel,
    )
