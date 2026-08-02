"""Physics backend interface for refinement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from refinement.problem import RefinementProblem


class PhysicsBackend(ABC):
    """Evaluate optical properties and thickness derivatives for a refinement problem."""

    @abstractmethod
    def evaluate_values(self, problem: RefinementProblem) -> list[float]:
        """Return flattened calculated values aligned with ``problem.all_target_values``."""

    @abstractmethod
    def evaluate_derivatives(self, problem: RefinementProblem) -> list[list[float]]:
        """Return Jacobian: ``[dY/da_j for each parameter j][residual index]``."""
