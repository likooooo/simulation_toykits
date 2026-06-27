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
    artifacts = Path(os.environ["SIMULATION_ARTIFACTS_DIR"]).resolve()
    expected_db = artifacts / "assets" / "database"
    actual_db = Path(os.environ.get("SIMULATION_DATABASE_DIR", "")).resolve()
    if actual_db != expected_db:
        pytest.fail(
            f"SIMULATION_DATABASE_DIR must be {expected_db} (collect path); got {actual_db}. "
            "Run: source scripts/init-toykits-build-env.sh"
        )
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
    import simulation_database_parser as sdp

    return sdp.get_simulation_database(init=True)


@pytest.fixture(scope="session")
def materials_db(sim_db):
    import simulation_database_parser as sdp

    from simulation_database.database_ui import object_catalog_name
    from toykits_config import DEFAULT_MATERIAL_PATH_KEYS

    out: dict[str, object] = {}
    for path_keys in DEFAULT_MATERIAL_PATH_KEYS:
        obj = sdp.read_at_query_path(sim_db, path_keys)
        catalog = object_catalog_name(obj)
        out[catalog] = obj
    return out


@pytest.fixture(scope="session")
def preset_parsed_layers(materials_db, filmstack_visualizer):
    """Preset id -> parsed formula layers (shared by preset and formula tests)."""
    from filmstack_simulation.presets import build_formula_for_preset, get_wl_mid_um
    from toykits_config import (
        FILMSTACK_PRESET_CATALOG,
        PRESETS,
        RECOMMENDED_SIM_WL_FROM_UM,
        RECOMMENDED_SIM_WL_TO_UM,
    )

    wl_mid = get_wl_mid_um(RECOMMENDED_SIM_WL_FROM_UM, RECOMMENDED_SIM_WL_TO_UM)
    return {
        preset.id: filmstack_visualizer.parse_filmstack_formula_v1(
            build_formula_for_preset(preset.id, FILMSTACK_PRESET_CATALOG, materials_db, wl_mid)
        )
        for preset in PRESETS
    }


@pytest.fixture
def mock_streamlit_session(monkeypatch) -> dict[str, object]:
    import streamlit as st

    state: dict[str, object] = {}
    monkeypatch.setattr(st, "session_state", state)
    return state


@pytest.fixture
def polarization_test_stack(materials_db):
    """Shared air/SiO2/Si stack for polarization mode tests."""
    from filmstack_simulation.simulation import resolve_stack

    formula = "air 0 SiO2 0.1 Si 0"
    materials, thicknesses_um = resolve_stack(formula, materials_db)
    return materials, thicknesses_um
