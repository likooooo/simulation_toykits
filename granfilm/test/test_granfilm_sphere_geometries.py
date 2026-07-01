"""Tests for Sphere film / 2film / thin_cap geometries."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from granfilm.common.baseline import (
    default_2film_baseline_path,
    default_aspnes_baseline_path,
    default_film_baseline_path,
    default_hexagonal_baseline_path,
    default_invariants_baseline_path,
    default_quadrupole_baseline_path,
    default_square_baseline_path,
    default_thin_cap_baseline_path,
    load_baseline,
)
from granfilm.common.constants import EPS_VACUUM
from granfilm.sphere_island.case import (
    default_2film_case,
    default_aspnes_case,
    default_film_case,
    default_hexagonal_case,
    default_invariants_case,
    default_quadrupole_case,
    default_square_case,
    default_thin_cap_case,
)
from granfilm.sphere_island.pipeline import run_granfilm_sphere
from granfilm.sphere_island.step0_init import step0_init
from granfilm.sphere_island.step4_geometry import polarizabilities_cap, surf_const_coef_film


@pytest.fixture(scope="session")
def film_case():
    return default_film_case()


@pytest.fixture(scope="session")
def twofilm_case():
    return default_2film_case()


@pytest.fixture(scope="session")
def thin_cap_case():
    return default_thin_cap_case()


@pytest.fixture(scope="session")
def quadrupole_case():
    return default_quadrupole_case()


@pytest.fixture(scope="session")
def square_case():
    return default_square_case()


@pytest.fixture(scope="session")
def hexagonal_case():
    return default_hexagonal_case()


@pytest.fixture(scope="session")
def invariants_case():
    return default_invariants_case()


@pytest.fixture(scope="session")
def aspnes_case():
    return default_aspnes_case()


@pytest.fixture(scope="session")
def materials_db(sim_db):
    from granfilm.common.materials import build_granfilm_materials_db

    return build_granfilm_materials_db(sim_db=sim_db)


def _skip_if_no_baseline(path: Path) -> None:
    if not path.is_file():
        pytest.skip(f"GranFilm baseline not found: {path}")


class TestFilmGeometry:
    def test_surf_const_formula(self, film_case, materials_db):
        state = step0_init(film_case, materials_db)
        chi = surf_const_coef_film(state, 0)
        e3 = state.eps_island[0]
        t = film_case.film_thickness1
        assert chi.gamma == pytest.approx((e3 - EPS_VACUUM) * t)
        assert chi.beta == pytest.approx((1.0 / EPS_VACUUM - 1.0 / e3) * t)

    def test_pipeline_finite_dr(self, film_case, materials_db):
        result = run_granfilm_sphere(film_case, materials_db, write_viz=False)
        assert np.all(np.isfinite(result.dr))
        assert result.geometry == "film"

    def test_pipeline_vs_baseline(self, film_case, materials_db):
        path = default_film_baseline_path()
        _skip_if_no_baseline(path)
        baseline = load_baseline(path)
        result = run_granfilm_sphere(film_case, materials_db, write_viz=False)
        py_on_base = np.interp(baseline.energy_ev, result.energy, result.dr)
        assert float(np.max(np.abs(py_on_base - baseline.value))) < 1e-5


class TestTwoFilmGeometry:
    def test_pipeline_finite_dr(self, twofilm_case, materials_db):
        result = run_granfilm_sphere(twofilm_case, materials_db, write_viz=False)
        assert np.all(np.isfinite(result.dr))
        assert result.geometry == "2film"

    def test_pipeline_vs_baseline(self, twofilm_case, materials_db):
        path = default_2film_baseline_path()
        _skip_if_no_baseline(path)
        baseline = load_baseline(path)
        result = run_granfilm_sphere(twofilm_case, materials_db, write_viz=False)
        py_on_base = np.interp(baseline.energy_ev, result.energy, result.dr)
        assert float(np.max(np.abs(py_on_base - baseline.value))) < 1e-5


class TestThinCapGeometry:
    def test_polarizability_cap(self, thin_cap_case, materials_db):
        state = step0_init(thin_cap_case, materials_db)
        alpha = polarizabilities_cap(state, 0)
        h = 1.0 - abs(thin_cap_case.tr)
        volume = np.pi * h**2 * (1.0 - h / 3.0)
        e3 = state.eps_island[0]
        assert alpha[0, 0] == pytest.approx(volume * (e3 - EPS_VACUUM))

    def test_pipeline_finite_dr(self, thin_cap_case, materials_db):
        result = run_granfilm_sphere(thin_cap_case, materials_db, write_viz=False)
        assert np.all(np.isfinite(result.dr))
        assert result.geometry == "thin_cap"

    def test_pipeline_vs_baseline(self, thin_cap_case, materials_db):
        path = default_thin_cap_baseline_path()
        _skip_if_no_baseline(path)
        baseline = load_baseline(path)
        result = run_granfilm_sphere(thin_cap_case, materials_db, write_viz=False)
        py_on_base = np.interp(baseline.energy_ev, result.energy, result.dr)
        assert float(np.max(np.abs(py_on_base - baseline.value))) < 1e-5


class TestQuadrupoleIsland:
    def test_pipeline_finite_dr(self, quadrupole_case, materials_db):
        result = run_granfilm_sphere(quadrupole_case, materials_db, write_viz=False)
        assert np.all(np.isfinite(result.dr))
        assert quadrupole_case.interaction.strip().lower() == "quadrupole"
        assert quadrupole_case.geometry.strip().lower() == "island"

    def test_pipeline_vs_baseline(self, quadrupole_case, materials_db):
        path = default_quadrupole_baseline_path()
        _skip_if_no_baseline(path)
        baseline = load_baseline(path)
        result = run_granfilm_sphere(quadrupole_case, materials_db, write_viz=False)
        py_on_base = np.interp(baseline.energy_ev, result.energy, result.dr)
        assert float(np.max(np.abs(py_on_base - baseline.value))) < 1e-5


@pytest.fixture(scope="session")
def island_below_case():
    from granfilm.sphere_island.case import default_island_below_case

    return default_island_below_case()


class TestIslandBelow:
    def test_step0_below_substrate(self, island_below_case, materials_db):
        state = step0_init(island_below_case, materials_db)
        assert island_below_case.tr < island_below_case.MPpos
        assert state.above is False

    def test_pipeline_finite_dr(self, island_below_case, materials_db):
        result = run_granfilm_sphere(island_below_case, materials_db, write_viz=False)
        assert np.all(np.isfinite(result.dr))
        assert result.geometry == "island"
        assert result.init.above is False

    def test_pipeline_vs_baseline(self, island_below_case, materials_db):
        from granfilm.common.baseline import default_below_baseline_path

        path = default_below_baseline_path()
        _skip_if_no_baseline(path)
        baseline = load_baseline(path)
        result = run_granfilm_sphere(island_below_case, materials_db, write_viz=False)
        py_on_base = np.interp(baseline.energy_ev, result.energy, result.dr)
        assert float(np.max(np.abs(py_on_base - baseline.value))) < 1e-5


class TestSquareLatticeIsland:
    def test_pipeline_finite_dr(self, square_case, materials_db):
        result = run_granfilm_sphere(square_case, materials_db, write_viz=False)
        assert np.all(np.isfinite(result.dr))
        assert square_case.network.strip().lower() == "square"
        assert square_case.geometry.strip().lower() == "island"

    def test_pipeline_vs_baseline(self, square_case, materials_db):
        path = default_square_baseline_path()
        _skip_if_no_baseline(path)
        baseline = load_baseline(path)
        result = run_granfilm_sphere(square_case, materials_db, write_viz=False)
        py_on_base = np.interp(baseline.energy_ev, result.energy, result.dr)
        assert float(np.max(np.abs(py_on_base - baseline.value))) < 1e-5


class TestHexagonalLatticeIsland:
    def test_pipeline_finite_dr(self, hexagonal_case, materials_db):
        result = run_granfilm_sphere(hexagonal_case, materials_db, write_viz=False)
        assert np.all(np.isfinite(result.dr))
        assert hexagonal_case.network.strip().lower() == "hexagonal"
        assert hexagonal_case.geometry.strip().lower() == "island"

    def test_pipeline_vs_baseline(self, hexagonal_case, materials_db):
        path = default_hexagonal_baseline_path()
        _skip_if_no_baseline(path)
        baseline = load_baseline(path)
        result = run_granfilm_sphere(hexagonal_case, materials_db, write_viz=False)
        py_on_base = np.interp(baseline.energy_ev, result.energy, result.dr)
        assert float(np.max(np.abs(py_on_base - baseline.value))) < 1e-5


class TestInvariantsFresnelIsland:
    def test_pipeline_finite_dr(self, invariants_case, materials_db):
        result = run_granfilm_sphere(invariants_case, materials_db, write_viz=False)
        assert np.all(np.isfinite(result.dr))
        assert invariants_case.fresnel.strip().lower() == "invariants"
        assert invariants_case.geometry.strip().lower() == "island"

    def test_pipeline_vs_baseline(self, invariants_case, materials_db):
        path = default_invariants_baseline_path()
        _skip_if_no_baseline(path)
        baseline = load_baseline(path)
        result = run_granfilm_sphere(invariants_case, materials_db, write_viz=False)
        py_on_base = np.interp(baseline.energy_ev, result.energy, result.dr)
        assert float(np.max(np.abs(py_on_base - baseline.value))) < 1e-5


class TestAspnesFresnelIsland:
    def test_pipeline_finite_dr(self, aspnes_case, materials_db):
        result = run_granfilm_sphere(aspnes_case, materials_db, write_viz=False)
        assert np.all(np.isfinite(result.dr))
        assert aspnes_case.fresnel.strip().lower() == "aspnes"
        assert aspnes_case.geometry.strip().lower() == "island"

    def test_pipeline_vs_baseline(self, aspnes_case, materials_db):
        path = default_aspnes_baseline_path()
        _skip_if_no_baseline(path)
        baseline = load_baseline(path)
        result = run_granfilm_sphere(aspnes_case, materials_db, write_viz=False)
        py_on_base = np.interp(baseline.energy_ev, result.energy, result.dr)
        assert float(np.max(np.abs(py_on_base - baseline.value))) < 1e-5


@pytest.mark.skipif(
    not (Path(os.environ.get("GRANFILM_DIR", "/home/like/repos/GranFilm-v1.0")) / "Dielectric" / "ag.nk").is_file(),
    reason="GranFilm Dielectric tree missing",
)
class TestFilmStep0Materials:
    def test_film_eps_island_vs_granfilm(self, film_case, materials_db):
        from granfilm.common.baseline import granfilm_dir
        from granfilm.common.sopra_dielectric import reference_from_granfilm_tree

        state = step0_init(film_case, materials_db)
        ref = reference_from_granfilm_tree(
            granfilm_dir() / "Dielectric",
            "ag",
            state.energy,
            geometry="film",
            R_nm=film_case.film_thickness1,
            mean_free_path=film_case.mean_free_path,
            surface_effects=film_case.surface_effects,
        )
        assert np.max(np.abs(state.eps_island - ref)) < 1e-5
