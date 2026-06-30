"""Host-level defaults for simulation_toykits (workspace paths, wavelength range)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import simulation  # noqa: F401 — must load before simulation_database_parser
import simulation_database_parser as sdp

from filmstack_simulation.filmstack_optimization.local_search.opt_config import (
    get_freehand_initial_formula,
    get_freehand_initial_preset_id,
)
from filmstack_simulation.template_types import FilmstackTemplate

DEFAULT_SPECTRUM_PATH = sdp.spectrum_query_keys("AM1.5G")

RECOMMENDED_SIM_WL_FROM_UM = 0.38
RECOMMENDED_SIM_WL_TO_UM = 0.78

_FILMSTACK_PRESET_CATALOG = None


def get_default_material_path_keys() -> list[list[str]]:
    from common import get_default_material_path_keys as _keys

    return _keys()


def get_required_default_material_names() -> frozenset[str]:
    from common import get_required_default_material_names as _names

    return _names()


def _filmstack_preset_catalog():
    global _FILMSTACK_PRESET_CATALOG
    if _FILMSTACK_PRESET_CATALOG is None:
        from common import build_filmstack_preset_catalog

        _FILMSTACK_PRESET_CATALOG = build_filmstack_preset_catalog()
    return _FILMSTACK_PRESET_CATALOG


def __getattr__(name: str):
    if name == "DEFAULT_MATERIAL_PATH_KEYS":
        return get_default_material_path_keys()
    if name == "REQUIRED_DEFAULT_MATERIAL_NAMES":
        return get_required_default_material_names()
    if name == "FILMSTACK_PRESET_CATALOG":
        return _filmstack_preset_catalog()
    if name == "PRESETS":
        return _filmstack_preset_catalog().presets
    if name == "DEFAULT_PRESET_ID":
        from template_config import default_preset_id

        return default_preset_id()
    raise AttributeError(name)


@dataclass(frozen=True)
class FilmstackInitialDefaults:
    preset_id: str
    formula: str
    wl_from_um: float
    wl_to_um: float


def resolve_filmstack_initial_defaults(
    valid_preset_ids: frozenset[str],
    *,
    template_by_id: Mapping[str, FilmstackTemplate] | None = None,
) -> FilmstackInitialDefaults:
    preset_id = get_freehand_initial_preset_id(valid_preset_ids)
    wl_from = RECOMMENDED_SIM_WL_FROM_UM
    wl_to = RECOMMENDED_SIM_WL_TO_UM
    if template_by_id and preset_id in template_by_id:
        sim = template_by_id[preset_id].sim
        if sim.wl_from_um is not None:
            wl_from = float(sim.wl_from_um)
        if sim.wl_to_um is not None:
            wl_to = float(sim.wl_to_um)
    return FilmstackInitialDefaults(
        preset_id=preset_id,
        formula=get_freehand_initial_formula(),
        wl_from_um=wl_from,
        wl_to_um=wl_to,
    )
