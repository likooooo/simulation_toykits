"""Refinement problem: stack parameters and flattened targets."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from refinement.target_spec import RefinementTargetSpec, inequality_to_lm

_CROSSCHECK = Path(__file__).resolve().parents[1] / "crosscheck"
if _CROSSCHECK.is_dir() and str(_CROSSCHECK) not in sys.path:
    sys.path.insert(0, str(_CROSSCHECK))

from stack_spec import StackSpec  # noqa: E402

from moremath.Levenberg_Marquardt import INFINITY  # noqa: E402


@dataclass(frozen=True)
class TargetSlice:
    start: int
    end: int
    target_idx: int


class RefinementProblem:
    """Film-thickness refinement problem over a ``StackSpec`` and target list."""

    def __init__(
        self,
        spec: StackSpec,
        materials_db: dict[str, Any],
        targets: Sequence[RefinementTargetSpec],
        refine_layer_indices: Sequence[int] | None = None,
    ) -> None:
        self.spec = spec
        self.materials_db = materials_db
        self.targets = list(targets)

        n_layers = len(spec.film_tokens)
        if refine_layer_indices is None:
            refine_layer_indices = list(range(n_layers))
        else:
            for idx in refine_layer_indices:
                if idx < 0 or idx >= n_layers:
                    raise ValueError(f"refine_layer_indices contains invalid layer index {idx}")

        self.refine_layer_indices = list(refine_layer_indices)
        self.parameters: list[tuple[int, float]] = [
            (layer_idx, float(spec.film_thicknesses_nm[layer_idx]))
            for layer_idx in self.refine_layer_indices
        ]

        self.a_min = [0.0] * len(self.parameters)
        self.a_max = [INFINITY] * len(self.parameters)

        self.target_slices: list[TargetSlice] = []
        self.all_target_values: list[float] = []
        self.all_tolerances: list[float] = []
        self.all_inequalities: list[int] = []

        start = 0
        for target_idx, target in enumerate(self.targets):
            n_points = len(target.values)
            end = start + n_points
            self.target_slices.append(TargetSlice(start, end, target_idx))
            self.all_target_values.extend(float(v) for v in target.values)
            self.all_tolerances.extend(float(t) for t in target.tolerances)
            lm_ineq = inequality_to_lm(target.inequality)
            self.all_inequalities.extend([lm_ineq] * n_points)
            start = end

    @property
    def parameter_layer_indices(self) -> list[int]:
        """Film layer index for each LM parameter (alias for refine_layer_indices)."""
        return list(self.refine_layer_indices)

    @property
    def nb_parameters(self) -> int:
        return len(self.parameters)

    @property
    def nb_residuals(self) -> int:
        return len(self.all_target_values)

    def get_parameters(self) -> list[float]:
        return [value for _, value in self.parameters]

    def set_parameters(self, values: Sequence[float]) -> None:
        if len(values) != len(self.parameters):
            raise ValueError(
                f"expected {len(self.parameters)} parameter values, got {len(values)}"
            )
        thicknesses = list(self.spec.film_thicknesses_nm)
        new_parameters: list[tuple[int, float]] = []
        for (layer_idx, _), value in zip(self.parameters, values):
            thickness = float(value)
            thicknesses[layer_idx] = thickness
            new_parameters.append((layer_idx, thickness))
        self.spec.film_thicknesses_nm = thicknesses
        self.parameters = new_parameters

    def apply_parameters(self, a: Sequence[float]) -> None:
        """Update ``spec.film_thicknesses_nm`` from the LM parameter vector."""
        self.set_parameters(a)

    def layer_index_for_parameter(self, param_idx: int) -> int:
        return self.parameters[param_idx][0]
