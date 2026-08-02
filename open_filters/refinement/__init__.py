"""OpenFilters-style LM thickness refinement core."""

from refinement.config import (
    ACCEPTABLE_CHI_2,
    MAX_ITERATIONS,
    MIN_CHI_2_CHANGE,
    MIN_GRADIENT,
)
from refinement.problem import RefinementProblem, TargetSlice
from refinement.session import RefinementHistoryEntry, RefinementResult, RefinementSession
from refinement.target_spec import RefinementTargetSpec, inequality_to_lm

__all__ = [
    "ACCEPTABLE_CHI_2",
    "MAX_ITERATIONS",
    "MIN_CHI_2_CHANGE",
    "MIN_GRADIENT",
    "RefinementHistoryEntry",
    "RefinementProblem",
    "RefinementResult",
    "RefinementSession",
    "RefinementTargetSpec",
    "TargetSlice",
    "inequality_to_lm",
]
