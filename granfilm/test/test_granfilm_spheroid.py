"""GranFilm oblate/prolate spheroid tests and baseline comparison."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from granfilm.common.baseline import (
    default_coated_baseline_path,
    default_prolate_baseline_path,
    default_spheroid_baseline_path,
    default_yamaguchi_baseline_path,
    load_baseline,
    parse_spheroid_test_dat,
)
from granfilm.common.materials import build_granfilm_materials_db
from granfilm.oblate_prolate.case import (
    default_coated_case,
    default_prolate_case,
    default_spheroid_case,
    default_yamaguchi_case,
)
from granfilm.oblate_prolate.pipeline import run_granfilm_spheroid
from granfilm.oblate_prolate.step0_init import step0_init
from granfilm.oblate_prolate.step1_integrals import step1_integrals
from granfilm.paths import OBLATE_COMPARE_DIR

PROLATE_COMPARE_DIR = OBLATE_COMPARE_DIR / "prolate"
YAMAGUCHI_COMPARE_DIR = OBLATE_COMPARE_DIR / "yamaguchi"
COATED_COMPARE_DIR = OBLATE_COMPARE_DIR / "coated"


@pytest.fixture(scope="session")
def spheroid_case():
    if os.environ.get("GRANFILM_DIR", "").strip():
        return default_spheroid_case(granfilm_dir=os.environ["GRANFILM_DIR"])
    return default_spheroid_case()


@pytest.fixture(scope="session")
def spheroid_materials_db(sim_db):
    return build_granfilm_materials_db(sim_db=sim_db)


@pytest.fixture(scope="session")
def spheroid_baseline():
    path = default_spheroid_baseline_path()
    if not path.is_file():
        pytest.skip(f"Spheroid baseline not found: {path}")
    return load_baseline(path)


@pytest.fixture(scope="session")
def prolate_case():
    return default_prolate_case()


@pytest.fixture(scope="session")
def prolate_baseline():
    path = default_prolate_baseline_path()
    if not path.is_file():
        pytest.skip(f"Prolate baseline not found: {path}")
    return load_baseline(path)


@pytest.fixture(scope="session")
def yamaguchi_case():
    return default_yamaguchi_case()


@pytest.fixture(scope="session")
def yamaguchi_baseline():
    path = default_yamaguchi_baseline_path()
    if not path.is_file():
        pytest.skip(f"Yamaguchi baseline not found: {path}")
    return load_baseline(path)


@pytest.fixture(scope="session")
def coated_case():
    return default_coated_case()


@pytest.fixture(scope="session")
def coated_baseline():
    path = default_coated_baseline_path()
    if not path.is_file():
        pytest.skip(f"Coated baseline not found: {path}")
    return load_baseline(path)


class TestSpheroidStep0:
    def test_oblate_derived_geometry(self, spheroid_case, spheroid_materials_db):
        state = step0_init(spheroid_case, spheroid_materials_db)
        assert state.island_type == "oblate"
        assert state.xi0 == pytest.approx(0.5773502691896257, rel=1e-6)
        assert state.a == pytest.approx(4.330127018922194, rel=1e-6)
        assert state.coverage == pytest.approx(spheroid_case.coverage, rel=1e-3)
        assert np.all(np.isfinite(state.eps_island))


class TestSpheroidStep1:
    def test_q000(self, spheroid_case, spheroid_materials_db):
        state = step0_init(spheroid_case, spheroid_materials_db)
        ints = step1_integrals(state, nint=spheroid_case.Nint)
        assert ints.Q[0, 0, 0] == pytest.approx(spheroid_case.tr + 1.0, rel=1e-4)


class TestSpheroidPipelineVsBaseline:
    def test_pipeline_finite_dr(
        self,
        spheroid_case,
        spheroid_materials_db,
        spheroid_baseline,
    ):
        result = run_granfilm_spheroid(
            spheroid_case,
            spheroid_materials_db,
            viz_dir=OBLATE_COMPARE_DIR,
            baseline=spheroid_baseline,
        )
        assert len(result.dr) == spheroid_case.Nenergy
        assert np.all(np.isfinite(result.dr))

        metrics_path = OBLATE_COMPARE_DIR / "metrics.json"
        assert metrics_path.is_file()
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert metrics["max_abs_err"] < 1e-5


class TestYamaguchiStep0:
    def test_yamaguchi_derived_geometry(self, yamaguchi_case, spheroid_materials_db):
        state = step0_init(yamaguchi_case, spheroid_materials_db)
        assert yamaguchi_case.geometry.strip().lower() == "yamaguchi"
        assert state.island_type == "oblate"
        assert state.xi0 == pytest.approx(0.5773502691896257, rel=1e-6)
        assert np.all(np.isfinite(state.eps_island))


class TestYamaguchiPipelineVsBaseline:
    def test_pipeline_finite_dr(
        self,
        yamaguchi_case,
        spheroid_materials_db,
        yamaguchi_baseline,
    ):
        result = run_granfilm_spheroid(
            yamaguchi_case,
            spheroid_materials_db,
            viz_dir=YAMAGUCHI_COMPARE_DIR,
            baseline=yamaguchi_baseline,
        )
        assert len(result.dr) == yamaguchi_case.Nenergy
        assert np.all(np.isfinite(result.dr))

        metrics_path = YAMAGUCHI_COMPARE_DIR / "metrics.json"
        assert metrics_path.is_file()
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert metrics["max_abs_err"] < 1e-5


class TestCoatedStep0:
    def test_coated_derived_geometry(self, coated_case, spheroid_materials_db):
        state = step0_init(coated_case, spheroid_materials_db)
        assert coated_case.geometry.strip().lower() == "coated"
        assert state.island_type == "oblate"
        assert state.eps_coating is not None
        assert np.all(np.isfinite(state.eps_coating))


class TestCoatedPipelineVsBaseline:
    def test_pipeline_finite_dr(
        self,
        coated_case,
        spheroid_materials_db,
        coated_baseline,
    ):
        result = run_granfilm_spheroid(
            coated_case,
            spheroid_materials_db,
            viz_dir=COATED_COMPARE_DIR,
            baseline=coated_baseline,
        )
        assert len(result.dr) == coated_case.Nenergy
        assert np.all(np.isfinite(result.dr))

        metrics_path = COATED_COMPARE_DIR / "metrics.json"
        assert metrics_path.is_file()
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert metrics["max_abs_err"] < 1e-5


@pytest.mark.skipif(
    not (
        Path(os.environ.get("GRANFILM_DIR", "/home/like/repos/GranFilm-v1.0"))
        / "testing"
        / "SpheroidTest.dat"
    ).is_file(),
    reason="SpheroidTest.dat missing",
)
class TestProlateStep0:
    def test_prolate_derived_geometry(self, prolate_case, spheroid_materials_db):
        state = step0_init(prolate_case, spheroid_materials_db)
        assert state.island_type == "prolate"
        assert state.xi0 == pytest.approx(1.1547005383792517, rel=1e-6)
        assert state.a == pytest.approx(4.330127018922194, rel=1e-6)
        assert state.coverage == pytest.approx(prolate_case.coverage, rel=1e-3)
        assert np.all(np.isfinite(state.eps_island))


class TestProlateStep1:
    def test_q000(self, prolate_case, spheroid_materials_db):
        state = step0_init(prolate_case, spheroid_materials_db)
        ints = step1_integrals(state, nint=prolate_case.Nint)
        assert ints.Q[0, 0, 0] == pytest.approx(prolate_case.tr + 1.0, rel=1e-4)


class TestProlatePipelineVsBaseline:
    def test_pipeline_finite_dr(
        self,
        prolate_case,
        spheroid_materials_db,
        prolate_baseline,
    ):
        result = run_granfilm_spheroid(
            prolate_case,
            spheroid_materials_db,
            viz_dir=PROLATE_COMPARE_DIR,
            baseline=prolate_baseline,
        )
        assert len(result.dr) == prolate_case.Nenergy
        assert np.all(np.isfinite(result.dr))

        metrics_path = PROLATE_COMPARE_DIR / "metrics.json"
        assert metrics_path.is_file()
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert metrics["max_abs_err"] < 1e-5


@pytest.mark.skipif(
    not default_prolate_baseline_path().is_file(),
    reason="ProlateTest.dat missing",
)
class TestProlateBaselineReference:
    def test_prolate_test_dat_finite(self):
        path = default_prolate_baseline_path()
        spec = parse_spheroid_test_dat(path)
        assert len(spec.energy_ev) >= 10
        assert np.all(np.isfinite(spec.value))


class TestSpheroidBaselineReference:
    def test_spheroid_test_dat_header_peak(self):
        path = default_spheroid_baseline_path()
        spec = parse_spheroid_test_dat(path)
        peak_i = int(np.argmax(spec.value))
        assert spec.energy_ev[peak_i] == pytest.approx(2.136, abs=0.05)
