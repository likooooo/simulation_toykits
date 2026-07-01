"""Simulation TMM adjoint physics backend for LM refinement."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from refinement.problem import RefinementProblem

try:
    from .base import PhysicsBackend
except ImportError:  # pragma: no cover - sibling module may not exist yet
    from abc import ABC, abstractmethod

    class PhysicsBackend(ABC):
        @abstractmethod
        def evaluate_values(self, problem: Any) -> list[float]:
            raise NotImplementedError

        @abstractmethod
        def evaluate_derivatives(self, problem: Any) -> list[list[float]]:
            raise NotImplementedError


def _ensure_crosscheck_on_path() -> None:
    crosscheck = Path(__file__).resolve().parents[2] / "crosscheck"
    path = str(crosscheck)
    if path not in sys.path:
        sys.path.insert(0, path)


_ensure_crosscheck_on_path()
from bootstrap_simulation import bootstrap_toykits_session  # noqa: E402
from simulation_derivatives import (  # noqa: E402
    simulation_rt_at,
    simulation_thickness_rt_derivatives,
)
from stack_spec import build_simulation_layers  # noqa: E402

_REFLECTANCE_KINDS = frozenset({"R", "R_spectrum"})
_TRANSMITTANCE_KINDS = frozenset({"T", "T_spectrum"})

# simulation adjoint returns dY/dd_um; LM parameters are thickness in nm.
# dY/dd_nm = dY/dd_um * (d_um/dd_nm) = dY/dd_um / 1000
_NM_PER_UM = 1000.0


def _target_uses_reflectance(kind: str) -> bool:
    if kind in _REFLECTANCE_KINDS:
        return True
    if kind in _TRANSMITTANCE_KINDS:
        return False
    raise ValueError(f"unsupported target kind for simulation backend: {kind!r}")


class SimulationBackend(PhysicsBackend):
    """Evaluate R/T spectra and thickness Jacobians via simulation adjoint."""

    def __init__(self) -> None:
        bootstrap_toykits_session()
        self._layers_cache_key: tuple[Any, ...] | None = None
        self._layers: list[Any] | None = None

    def _layers_for(self, problem: RefinementProblem) -> list[Any]:
        key = (
            tuple(problem.spec.film_tokens),
            tuple(problem.spec.film_thicknesses_nm),
            problem.spec.incident_token,
            problem.spec.substrate_token,
            id(problem.materials_db),
        )
        if key != self._layers_cache_key:
            self._layers = build_simulation_layers(problem.materials_db, problem.spec)
            self._layers_cache_key = key
        assert self._layers is not None
        return self._layers

    def evaluate_values(self, problem: RefinementProblem) -> list[float]:
        """Return flattened model values Y[i] aligned with problem.target_slices."""
        layers = self._layers_for(problem)
        values = [0.0] * len(problem.all_target_values)

        for sl in problem.target_slices:
            start, end, target_idx = sl.start, sl.end, sl.target_idx
            target = problem.targets[target_idx]
            use_r = _target_uses_reflectance(target.kind)
            for local_i, global_i in enumerate(range(start, end)):
                wl_nm = target.wavelengths_nm[local_i]
                r_val, t_val = simulation_rt_at(
                    layers,
                    wl_nm / _NM_PER_UM,
                    target.angle_deg,
                    target.polarization,
                )
                values[global_i] = float(r_val if use_r else t_val)

        return values

    def evaluate_derivatives(self, problem: RefinementProblem) -> list[list[float]]:
        """Return dY/da_nm with shape [nb_parameters][nb_points]."""
        nb_points = len(problem.all_target_values)
        all_derivatives = [[0.0] * nb_points for _ in range(problem.nb_parameters)]

        for sl in problem.target_slices:
            start, end, target_idx = sl.start, sl.end, sl.target_idx
            target = problem.targets[target_idx]
            dR, dT = simulation_thickness_rt_derivatives(
                problem.spec,
                problem.materials_db,
                target.wavelengths_nm,
                target.angle_deg,
                target.polarization,
            )
            mat = dR if _target_uses_reflectance(target.kind) else dT
            for local_i, global_i in enumerate(range(start, end)):
                for par, layer_idx in enumerate(problem.parameter_layer_indices):
                    sim_layer = 1 + layer_idx
                    all_derivatives[par][global_i] = (
                        float(mat[sim_layer, local_i]) / _NM_PER_UM
                    )

        return all_derivatives
