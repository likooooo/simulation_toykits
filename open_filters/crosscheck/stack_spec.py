"""Shared film stack definition for OpenFilters vs simulation cross-check."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from materials_db import OF_FILMSTACK_MATERIAL_PATHS


@dataclass
class StackSpec:
    """Air | films... | substrate aligned between simulation and OpenFilters."""

    incident_token: str = "of_void"
    film_tokens: list[str] = field(default_factory=lambda: ["of_TiO2", "of_SiO2"])
    film_thicknesses_nm: list[float] = field(default_factory=lambda: [100.0, 50.0])
    substrate_token: str = "of_BK7"

    def film_thicknesses_um(self) -> list[float]:
        return [t / 1000.0 for t in self.film_thicknesses_nm]

    def simulation_layer_tokens(self) -> list[tuple[str, float]]:
        """[(material_token, depth_um), ...] including bookends."""
        out: list[tuple[str, float]] = [(self.incident_token, 0.0)]
        for tok, t_nm in zip(self.film_tokens, self.film_thicknesses_nm):
            out.append((tok, t_nm / 1000.0))
        out.append((self.substrate_token, 0.0))
        return out

    def openfilters_layer_count(self) -> int:
        return len(self.film_tokens)

    def simulation_film_layer_indices(self) -> list[int]:
        """Layer indices in simulation stack with finite thickness."""
        return list(range(1, 1 + len(self.film_tokens)))


DEFAULT_BRAGG_STACK = StackSpec()


def nk_at_wavelength_um(material: Any, wl_um: float) -> complex:
    nk = material.nk_at_wavelength_um(float(wl_um))
    if isinstance(nk, complex):
        return nk
    return complex(nk)


def simulation_nk_to_abeles(nk: complex) -> complex:
    """Map simulation (n + i*k_abs) to OpenFilters abeles (n - i*k_abs)."""
    return complex(nk.real, -abs(nk.imag))


def sample_stack_nk(
    materials_db: dict[str, Any],
    spec: StackSpec,
    wls_nm: Sequence[float],
) -> tuple[list[list[complex]], complex, complex]:
    """Return per-layer N[layer][iwl], medium_nk, substrate_nk at each wl."""
    wls_um = [float(w) / 1000.0 for w in wls_nm]
    layers_nk: list[list[complex]] = []
    for tok in spec.film_tokens:
        mat = materials_db[tok]
        layers_nk.append([nk_at_wavelength_um(mat, w) for w in wls_um])
    medium = [nk_at_wavelength_um(materials_db[spec.incident_token], w) for w in wls_um]
    substrate = [nk_at_wavelength_um(materials_db[spec.substrate_token], w) for w in wls_um]
    return layers_nk, medium, substrate


def build_simulation_layers(materials_db: dict[str, Any], spec: StackSpec) -> list[Any]:
    from filmstack_visualizer import build_tmm_layers

    mats = [materials_db[tok] for tok, _ in spec.simulation_layer_tokens()]
    depths = [d for _, d in spec.simulation_layer_tokens()]
    return build_tmm_layers(mats, depths)


def load_default_materials_db(sim_db: Any | None = None) -> dict[str, Any]:
    from simulation_database_parser import get_simulation_database, materials_db_from_token_paths

    db = sim_db or get_simulation_database(init=True)
    return materials_db_from_token_paths(OF_FILMSTACK_MATERIAL_PATHS, sim_db=db)
