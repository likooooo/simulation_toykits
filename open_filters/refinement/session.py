"""Levenberg-Marquardt refinement session."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from moremath import Levenberg_Marquardt

from refinement.backends.base import PhysicsBackend
from refinement.config import (
    ACCEPTABLE_CHI_2,
    MAX_ITERATIONS,
    MIN_CHI_2_CHANGE,
    MIN_GRADIENT,
)
from refinement.problem import RefinementProblem


@dataclass
class RefinementHistoryEntry:
    iteration: int
    chi_2: float
    thicknesses_nm: list[float]
    status: int


@dataclass
class RefinementResult:
    status: int
    chi_2: float
    iterations: int
    thicknesses_nm: list[float]
    parameters: list[float]
    history: list[RefinementHistoryEntry] = field(default_factory=list)
    stop_reason: str = ""


_STOP_REASONS = {
    Levenberg_Marquardt.IMPROVING: "improving",
    Levenberg_Marquardt.MINIMUM_FOUND: "minimum_found",
    Levenberg_Marquardt.CHI_2_IS_OK: "chi_2_is_ok",
    Levenberg_Marquardt.CHI_2_CHANGE_TOO_SMALL: "chi_2_change_too_small",
    Levenberg_Marquardt.DELTA_IS_TOO_SMALL: "delta_is_too_small",
    Levenberg_Marquardt.ALL_PARAMETERS_ARE_STUCK: "all_parameters_are_stuck",
    Levenberg_Marquardt.SINGULAR_MATRIX: "singular_matrix",
}


class RefinementSession:
    """Run thickness refinement via vendored OpenFilters Levenberg-Marquardt."""

    def __init__(
        self,
        problem: RefinementProblem,
        backend: PhysicsBackend,
        a0: list[float] | None = None,
        *,
        min_gradient: float = MIN_GRADIENT,
        acceptable_chi_2: float = ACCEPTABLE_CHI_2,
        min_chi_2_change: float = MIN_CHI_2_CHANGE,
    ) -> None:
        self.problem = problem
        self.backend = backend
        self.min_gradient = min_gradient
        self.acceptable_chi_2 = acceptable_chi_2
        self.min_chi_2_change = min_chi_2_change

        if a0 is None:
            a0 = problem.get_parameters()
        self.problem.set_parameters(a0)

        self.optimizer = Levenberg_Marquardt.Levenberg_Marquardt(
            self.calculate_values,
            self.calculate_derivatives,
            self.problem.get_parameters(),
            self.problem.all_target_values,
            self.problem.all_tolerances,
        )
        self.optimizer.set_stop_criteria(
            self.min_gradient,
            self.acceptable_chi_2,
            self.min_chi_2_change,
        )
        self.optimizer.set_limits(self.problem.a_min, self.problem.a_max)
        self.optimizer.set_inequalities(self.problem.all_inequalities)

        self.status = Levenberg_Marquardt.IMPROVING
        self.chi_2 = 0.0
        self.iteration = 0
        self.history: list[RefinementHistoryEntry] = []

    @property
    def thicknesses_nm(self) -> list[float]:
        return list(self.problem.spec.film_thicknesses_nm)

    def calculate_values(self, a: list[float], *_args: Any) -> list[float]:
        self.problem.apply_parameters(a)
        return self.backend.evaluate_values(self.problem)

    def calculate_derivatives(self, a: list[float], *_args: Any) -> list[list[float]]:
        self.problem.apply_parameters(a)
        return self.backend.evaluate_derivatives(self.problem)

    def _record_history(self) -> None:
        self.history.append(
            RefinementHistoryEntry(
                iteration=self.iteration,
                chi_2=self.chi_2,
                thicknesses_nm=self.thicknesses_nm,
                status=self.status,
            )
        )

    def prepare(self) -> None:
        """Initialize LM state and record starting chi_2."""
        self.optimizer.prepare()
        self.status = Levenberg_Marquardt.IMPROVING
        self.chi_2 = self.optimizer.get_chi_2()
        self.problem.set_parameters(self.optimizer.a)
        self._record_history()

    def iterate(self) -> int:
        """Perform one LM iteration; return LM status code."""
        self.status = self.optimizer.iterate()
        self.chi_2 = self.optimizer.get_chi_2()
        self.problem.set_parameters(self.optimizer.a)
        self.iteration += 1
        self._record_history()
        return self.status

    def run(self, max_iter: int | None = MAX_ITERATIONS) -> RefinementResult:
        """Run ``prepare`` then iterate until stop or ``max_iter``."""
        self.prepare()
        while True:
            if max_iter is not None and self.iteration >= max_iter:
                break
            status = self.iterate()
            if status != Levenberg_Marquardt.IMPROVING:
                break

        stop_reason = _STOP_REASONS.get(self.status, f"unknown_status_{self.status}")
        if max_iter is not None and self.iteration >= max_iter and self.status == Levenberg_Marquardt.IMPROVING:
            stop_reason = "max_iterations"

        return RefinementResult(
            status=self.status,
            chi_2=self.chi_2,
            iterations=self.iteration,
            thicknesses_nm=self.thicknesses_nm,
            parameters=self.problem.get_parameters(),
            history=list(self.history),
            stop_reason=stop_reason,
        )
