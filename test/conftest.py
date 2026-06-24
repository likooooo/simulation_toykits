"""Shared pytest fixtures for simulation_toykits tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

STANDARD_AR_FORMULA = (
    "air 0 (SiO2 0.1 1.46 0 Ta2O5 0.02 2.13 0.001)^2 Si 0 3.87 0.02"
)


def pytest_configure(config) -> None:
    if not os.environ.get("SIMULATION_ARTIFACTS_DIR", "").strip():
        pytest.fail("source scripts/init-toykits-build-env.sh before pytest")
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    os.chdir(os.environ["SIMULATION_ARTIFACTS_DIR"])
    import simulation  # noqa: F401


@pytest.fixture(scope="session")
def simulation():
    import simulation as sim

    return sim


@pytest.fixture(scope="session")
def filmstack_visualizer(simulation):
    import filmstack_visualizer as fv

    return fv


@pytest.fixture(scope="session")
def standard_ar_formula() -> str:
    return STANDARD_AR_FORMULA


@pytest.fixture(scope="session")
def sample_tmm_layers(standard_ar_formula):
    from filmstack_simulation.simulation import resolve_stack_with_layers

    _, _, layers = resolve_stack_with_layers(standard_ar_formula, {})
    return layers


@pytest.fixture(scope="session")
def sim_db():
    from simulation_database.database_ui import ensure_simulation_database_initialized

    return ensure_simulation_database_initialized()


@pytest.fixture(scope="session")
def materials_db(sim_db):
    return load_materials_from_paths(sim_db)


@pytest.fixture(scope="session")
def preset_parsed_layers(materials_db, filmstack_visualizer):
    """Preset id -> parsed formula layers (shared by preset and formula tests)."""
    from filmstack_simulation.materials import RECOMMENDED_SIM_WL_FROM_UM, RECOMMENDED_SIM_WL_TO_UM
    from filmstack_simulation.presets import PRESETS, build_formula_for_preset, get_wl_mid_um

    wl_mid = get_wl_mid_um(RECOMMENDED_SIM_WL_FROM_UM, RECOMMENDED_SIM_WL_TO_UM)
    return {
        preset.id: filmstack_visualizer.parse_filmstack_formula_v1(
            build_formula_for_preset(preset.id, materials_db, wl_mid)
        )
        for preset in PRESETS
    }


def load_materials_from_paths(sim_db, path_keys=None) -> dict[str, object]:
    from filmstack_simulation.materials import DEFAULT_MATERIAL_PATH_KEYS
    from simulation_database.database_ui import object_catalog_name, read_leaf_at_path

    keys = path_keys if path_keys is not None else DEFAULT_MATERIAL_PATH_KEYS
    db: dict[str, object] = {}
    for path in keys:
        obj = read_leaf_at_path(sim_db, "materials", path)
        db[object_catalog_name(obj)] = obj
    return db
