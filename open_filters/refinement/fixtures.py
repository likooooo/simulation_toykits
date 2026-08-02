"""Shared parity test stack and iteration settings."""

from __future__ import annotations

from stack_spec import StackSpec

# LM outer loop iterations for abeles vs simulation parity tests/plots.
PARITY_MAX_ITER = 30

# Thickness trajectory match tolerance (nm) per LM iteration.
PARITY_DA_TOL_NM = 0.05

# Two-film stack: simulation adjoint and OpenFilters abeles derivatives align (~1e-6 rel).
# Used for honest LM parity (each backend uses its own f and df).
PARITY_REFINEMENT_STACK = StackSpec(
    film_tokens=["of_TiO2", "of_SiO2"],
    film_thicknesses_nm=[58.0, 95.0],
)

# Six-film stack for forward/complexity and LM parity scenarios.
# Thickness adjoint suffix multiply order fixed in transfer_matrix_method_adjoint.hpp.
PARITY_COMPLEX_STACK = StackSpec(
    film_tokens=[
        "of_TiO2",
        "of_SiO2",
        "of_TiO2",
        "of_SiO2",
        "of_TiO2",
        "of_SiO2",
    ],
    film_thicknesses_nm=[58.0, 95.0, 58.0, 95.0, 58.0, 95.0],
)

PARITY_SPECTRUM_WLS_NM = [
    400.0,
    430.0,
    460.0,
    490.0,
    520.0,
    550.0,
    580.0,
    610.0,
    640.0,
    670.0,
    700.0,
]
