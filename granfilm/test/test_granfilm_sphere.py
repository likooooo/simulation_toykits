"""GranFilm Sphere / island step tests and baseline comparison."""

from __future__ import annotations

import json

import numpy as np
import pytest

from granfilm.common.baseline import (
    default_baseline_path,
    granfilm_dir,
    load_baseline,
    parse_sphere_test_dat,
)
from granfilm.common.materials import build_granfilm_materials_db, epsilon_grid
from granfilm.common.zeta import step1_zeta
from granfilm.paths import SPHERE_COMPARE_DIR
from granfilm.sphere_island.case import default_sphere_case
from granfilm.sphere_island.pipeline import run_granfilm_sphere
from granfilm.sphere_island.step0_init import step0_init
from granfilm.sphere_island.step1_integrals import step1_integrals

VIZ_DIR = SPHERE_COMPARE_DIR


@pytest.fixture(scope="session")
def granfilm_case():
    return default_sphere_case(granfilm_dir=str(granfilm_dir()))


@pytest.fixture(scope="session")
def granfilm_materials_db(sim_db):
    return build_granfilm_materials_db(sim_db=sim_db)


@pytest.fixture(scope="session")
def baseline_spectrum():
    path = default_baseline_path()
    if not path.is_file():
        pytest.skip(f"GranFilm baseline not found: {path}")
    return load_baseline(path)


class TestStep1Zeta:
    def test_zeta_m0_diagonal(self, granfilm_case):
        z = step1_zeta(granfilm_case.Mpole_order, m_max=1)
        l1, l2 = 3, 5
        expected = 0.5 * np.sqrt((2 * l1 + 1) * (2 * l2 + 1))
        assert z[l1, l2, 0] == pytest.approx(expected)

    def test_zeta_m1_l1(self, granfilm_case):
        z = step1_zeta(granfilm_case.Mpole_order, m_max=1)
        l1 = l2 = 2
        fac = 1.0 / (l1 * (l1 + 1) * l2 * (l2 + 1))
        expected = 0.5 * np.sqrt((2 * l1 + 1) * (2 * l2 + 1) * fac)
        assert z[l1, l2, 1] == pytest.approx(expected)


class TestStep0Materials:
    def test_ag_mgo_epsilon_finite(self, granfilm_case, granfilm_materials_db):
        state = step0_init(granfilm_case, granfilm_materials_db)
        assert np.all(np.isfinite(state.eps_island))
        assert np.all(np.isfinite(state.eps_substrate))
        assert state.density > 0
        assert state.coverage == pytest.approx(granfilm_case.coverage, rel=1e-3)

    def test_epsilon_grid_shape(self, granfilm_materials_db):
        e = np.linspace(1.5, 4.5, 10)
        eps = epsilon_grid(granfilm_materials_db, "ag", e, mean_free_path="finite_size", surface_effects=True)
        assert eps.shape == (10,)
        assert np.iscomplexobj(eps)

    def test_gf_materials_match_granfilm_nk(self, granfilm_case, granfilm_materials_db):
        """Step 0: gf YAML pipeline vs GranFilm Dielectric/*.nk (Fortran algorithm)."""
        from granfilm.common.sopra_dielectric import reference_from_granfilm_tree

        energy = np.linspace(
            granfilm_case.energy_min,
            granfilm_case.energy_max,
            granfilm_case.Nenergy,
            dtype=np.float64,
        )
        dielectric = granfilm_dir() / "Dielectric"

        state = step0_init(granfilm_case, granfilm_materials_db)
        ref_ag = reference_from_granfilm_tree(
            dielectric,
            "ag",
            energy,
            geometry=granfilm_case.geometry,
            tr=granfilm_case.tr,
            R_nm=granfilm_case.R,
            mean_free_path=granfilm_case.mean_free_path,
            surface_effects=granfilm_case.surface_effects,
        )
        ref_mgo = reference_from_granfilm_tree(
            dielectric,
            "mgo",
            energy,
            mean_free_path="none",
            surface_effects=False,
        )
        assert np.max(np.abs(state.eps_island - ref_ag)) < 1e-5
        assert np.max(np.abs(state.eps_substrate - ref_mgo)) < 1e-5


class TestStep1Integrals:
    def test_q000(self, granfilm_case):
        Int = step1_integrals(
            granfilm_case.tr,
            granfilm_case.MPpos,
            granfilm_case.Mpole_order,
            m_max=1,
            nint=granfilm_case.Nint,
        )
        assert Int.Q[0, 0, 0] == pytest.approx(granfilm_case.tr + 1.0)

    def test_integrals_match_fortran_formula(self, granfilm_case):
        from granfilm.sphere_island.bisect_align import step1_integrals_with_quadrature

        cur = step1_integrals(
            granfilm_case.tr,
            granfilm_case.MPpos,
            granfilm_case.Mpole_order,
            m_max=1,
            nint=granfilm_case.Nint,
        )
        ref = step1_integrals_with_quadrature(
            granfilm_case.tr,
            granfilm_case.MPpos,
            granfilm_case.Mpole_order,
            m_max=1,
            nint=granfilm_case.Nint,
            quadrature="gauleg",
            fix_l0_ln=True,
        )
        for name in ("Q", "K", "L", "M", "N"):
            assert np.max(np.abs(getattr(cur, name) - getattr(ref, name))) < 1e-5


class TestPipelineVsBaseline:
    def test_pipeline_finite_dr(
        self,
        granfilm_case,
        granfilm_materials_db,
        baseline_spectrum,
    ):
        result = run_granfilm_sphere(
            granfilm_case,
            granfilm_materials_db,
            viz_dir=VIZ_DIR,
            write_viz=True,
        )
        assert len(result.dr) == granfilm_case.Nenergy
        assert np.all(np.isfinite(result.dr))

        from granfilm.sphere_island.viz import plot_step5_dr_compare

        metrics = plot_step5_dr_compare(
            result,
            baseline_spectrum,
            VIZ_DIR / "compare.png",
            metrics_path=VIZ_DIR / "metrics.json",
        )
        assert "rmse" in metrics
        assert metrics["rmse"] >= 0.0
        assert metrics["max_abs_err"] < 1e-5

        metrics_path = VIZ_DIR / "metrics.json"
        assert metrics_path.is_file()
        loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert loaded["note"]

    def test_baseline_dat_parseable(self, baseline_spectrum):
        assert len(baseline_spectrum.energy_ev) >= 50
        assert baseline_spectrum.value.shape == baseline_spectrum.energy_ev.shape


@pytest.mark.skipif(
    not (granfilm_dir() / "testing" / "SphereTest.dat").is_file(),
    reason="SphereTest.dat missing",
)
class TestBaselineReference:
    def test_sphere_test_dat_header_peak(self):
        path = default_baseline_path()
        spec = parse_sphere_test_dat(path)
        peak_i = int(np.argmax(spec.value))
        assert spec.energy_ev[peak_i] == pytest.approx(2.924, abs=0.05)
