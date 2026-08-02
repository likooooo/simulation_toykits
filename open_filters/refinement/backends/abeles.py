"""Abeles (OpenFilters) physics backend for LM refinement."""

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
from openfilters_derivatives import (  # noqa: E402
    POL_TE,
    POL_TM,
    POL_UNPOLARIZED,
    openfilters_rt_spectrum,
    openfilters_thickness_rt_derivatives,
)

_POLARIZATION_TO_ABELES: dict[str, float] = {
    "TE": POL_TE,
    "TM": POL_TM,
    "UNPOLARIZED": POL_UNPOLARIZED,
}

_REFLECTANCE_KINDS = frozenset({"R", "R_spectrum"})
_TRANSMITTANCE_KINDS = frozenset({"T", "T_spectrum"})


def _abeles_polarization(polarization: str) -> float:
    try:
        return _POLARIZATION_TO_ABELES[polarization]
    except KeyError as exc:
        raise ValueError(f"unsupported polarization for abeles backend: {polarization!r}") from exc


def _target_uses_reflectance(kind: str) -> bool:
    if kind in _REFLECTANCE_KINDS:
        return True
    if kind in _TRANSMITTANCE_KINDS:
        return False
    raise ValueError(f"unsupported target kind for abeles backend: {kind!r}")


class AbelesBackend(PhysicsBackend):
    """Evaluate R/T spectra and thickness Jacobians via OpenFilters abeles."""

    def __init__(self, *, of_root: Path | str | None = None) -> None:
        self.of_root = of_root

    def evaluate_values(self, problem: RefinementProblem) -> list[float]:
        """Return flattened model values Y[i] aligned with problem.target_slices."""
        values = [0.0] * len(problem.all_target_values)
        for sl in problem.target_slices:
            start, end, target_idx = sl.start, sl.end, sl.target_idx
            target = problem.targets[target_idx]
            pol = _abeles_polarization(target.polarization)
            r_spec, t_spec = openfilters_rt_spectrum(
                problem.spec,
                problem.materials_db,
                target.wavelengths_nm,
                target.angle_deg,
                pol,
                of_root=self.of_root,
            )
            spectrum = r_spec if _target_uses_reflectance(target.kind) else t_spec
            for local_i, global_i in enumerate(range(start, end)):
                values[global_i] = float(spectrum[local_i])
        return values

    def evaluate_derivatives(self, problem: RefinementProblem) -> list[list[float]]:
        """Return dY/da_nm with shape [nb_parameters][nb_points]."""
        nb_points = len(problem.all_target_values)
        all_derivatives = [[0.0] * nb_points for _ in range(problem.nb_parameters)]

        for sl in problem.target_slices:
            start, end, target_idx = sl.start, sl.end, sl.target_idx
            target = problem.targets[target_idx]
            pol = _abeles_polarization(target.polarization)
            dR, dT = openfilters_thickness_rt_derivatives(
                problem.spec,
                problem.materials_db,
                target.wavelengths_nm,
                target.angle_deg,
                pol,
                of_root=self.of_root,
            )
            mat = dR if _target_uses_reflectance(target.kind) else dT
            for local_i, global_i in enumerate(range(start, end)):
                for par, layer_idx in enumerate(problem.parameter_layer_indices):
                    all_derivatives[par][global_i] = float(mat[layer_idx, local_i])

        return all_derivatives
