"""Shared pytest fixtures for simulation_toykits tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PRECOMPILED_BUNDLE = "database.bin"

STANDARD_AR_FORMULA = (
    "air_Ciddor 0 1.0 0 (SiO2_Arosa 0.1 1.46 0 Ta2O5_Cheikh-amorphous-3.28-8-450 0.02 2.13 0.001)^2 Si_Aspnes 0 3.87 0.02"
)


def pytest_configure(config) -> None:
    if not os.environ.get("SIMULATION_ARTIFACTS_DIR", "").strip():
        pytest.fail("source scripts/init-toykits-build-env.sh before pytest")
    artifacts = Path(os.environ["SIMULATION_ARTIFACTS_DIR"]).resolve()
    expected_assets = artifacts / "assets"
    expected_bundle = expected_assets / _PRECOMPILED_BUNDLE
    actual_db = Path(os.environ.get("SIMULATION_DATABASE_DIR", "")).resolve()
    if actual_db != expected_assets:
        pytest.fail(
            f"SIMULATION_DATABASE_DIR must be {expected_assets} (artifact assets path); got {actual_db}. "
            "Run: source scripts/init-toykits-build-env.sh"
        )
    if not expected_bundle.is_file():
        pytest.fail(
            f"missing precompiled database bundle: {expected_bundle}. "
            "Run: python scripts/build_toykits.py"
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
    from common import build_default_materials_db

    return build_default_materials_db(sim_db=sim_db)


@pytest.fixture(scope="session")
def preset_parsed_layers(materials_db, filmstack_visualizer):
    """Preset id -> parsed formula layers (shared by preset and formula tests)."""
    from common import build_filmstack_preset_catalog

    catalog = build_filmstack_preset_catalog()
    return {
        preset.id: filmstack_visualizer.parse_filmstack_formula_v1(preset.formula)
        for preset in catalog.presets
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

    formula = "air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0"
    materials, thicknesses_um = resolve_stack(formula, materials_db)
    return materials, thicknesses_um
