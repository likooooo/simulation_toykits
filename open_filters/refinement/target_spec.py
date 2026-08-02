"""Refinement target description for LM optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TargetKind = Literal["R", "T", "R_spectrum", "T_spectrum"]
Polarization = Literal["TE", "TM", "UNPOLARIZED"]
Inequality = Literal["equal", "smaller", "larger"]


@dataclass
class RefinementTargetSpec:
    kind: TargetKind
    wavelengths_nm: list[float]
    values: list[float]
    tolerances: list[float]
    angle_deg: float = 0.0
    polarization: Polarization = "UNPOLARIZED"
    inequality: Inequality = "equal"
    consider_backside: bool = False

    def __post_init__(self) -> None:
        n_wl = len(self.wavelengths_nm)
        n_val = len(self.values)
        n_tol = len(self.tolerances)
        if n_wl != n_val or n_wl != n_tol:
            raise ValueError(
                "wavelengths_nm, values, and tolerances must have the same length "
                f"(got {n_wl}, {n_val}, {n_tol})"
            )
        if self.kind in ("R", "T") and n_wl != 1:
            raise ValueError(f"{self.kind} targets require exactly one wavelength/value pair")


def inequality_to_lm(inequality: Inequality) -> int:
    """Map target inequality to Levenberg_Marquardt SMALLER/EQUAL/LARGER (-1/0/1)."""
    return {"smaller": -1, "equal": 0, "larger": 1}[inequality]
