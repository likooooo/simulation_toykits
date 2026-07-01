"""Pytest configuration for granfilm tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from granfilm.paths import REPO_ROOT

_PRECOMPILED_BUNDLE = "database.bin"


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
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    os.chdir(os.environ["SIMULATION_ARTIFACTS_DIR"])
    import simulation  # noqa: F401


@pytest.fixture(scope="session")
def sim_db():
    import simulation_database_parser as sdp

    return sdp.get_simulation_database(init=True)
