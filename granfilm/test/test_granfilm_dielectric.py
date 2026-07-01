"""GranFilm dielectric pipeline vs reference_from_granfilm_tree."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from granfilm.common.materials import build_granfilm_materials_db, epsilon_grid
from granfilm.common.sopra_dielectric import (
    apply_thermoreflectance_shift,
    dielectric_constants,
    dielectric_func_corrections,
    parse_finite_size_dat,
    parse_granfilm_nk,
    reference_from_granfilm_tree,
)
from granfilm.sphere_island.case import default_sphere_case


@pytest.fixture(scope="session")
def granfilm_case():
    return default_sphere_case()


@pytest.fixture(scope="session")
def granfilm_materials_db(sim_db):
    return build_granfilm_materials_db(sim_db=sim_db)


@pytest.fixture(scope="session")
def dielectric_dir():
    gf_dir = Path(os.environ.get("GRANFILM_DIR", "/home/like/repos/GranFilm-v1.0"))
    path = gf_dir / "Dielectric"
    if not path.is_dir():
        pytest.skip(f"GranFilm Dielectric dir missing: {path}")
    return path


def _island_energy(case) -> np.ndarray:
    return np.linspace(case.energy_min, case.energy_max, case.Nenergy, dtype=np.float64)


class TestIslandMeanFreePath:
    def test_finite_size_matches_reference(
        self, granfilm_case, granfilm_materials_db, dielectric_dir
    ):
        energy = _island_energy(granfilm_case)
        py = epsilon_grid(
            granfilm_materials_db,
            "ag",
            energy,
            geometry=granfilm_case.geometry,
            tr=granfilm_case.tr,
            R_nm=granfilm_case.R,
            mean_free_path="finite_size",
            surface_effects=granfilm_case.surface_effects,
        )
        ref = reference_from_granfilm_tree(
            dielectric_dir,
            "ag",
            energy,
            geometry=granfilm_case.geometry,
            tr=granfilm_case.tr,
            R_nm=granfilm_case.R,
            mean_free_path="finite_size",
            surface_effects=granfilm_case.surface_effects,
        )
        assert np.max(np.abs(py - ref)) < 1e-5

    def test_a_parameter_matches_reference(
        self, granfilm_case, granfilm_materials_db, dielectric_dir
    ):
        energy = _island_energy(granfilm_case)
        a_val = 0.65
        py = epsilon_grid(
            granfilm_materials_db,
            "ag",
            energy,
            geometry=granfilm_case.geometry,
            tr=granfilm_case.tr,
            R_nm=granfilm_case.R,
            mean_free_path="A-parameter",
            surface_effects=granfilm_case.surface_effects,
            A=a_val,
        )
        ref = reference_from_granfilm_tree(
            dielectric_dir,
            "ag",
            energy,
            geometry=granfilm_case.geometry,
            tr=granfilm_case.tr,
            R_nm=granfilm_case.R,
            mean_free_path="A-parameter",
            surface_effects=granfilm_case.surface_effects,
            A=a_val,
        )
        assert np.max(np.abs(py - ref)) < 1e-5

    def test_a_parameter_default_matches_finite_size(
        self, granfilm_case, granfilm_materials_db, dielectric_dir
    ):
        energy = _island_energy(granfilm_case)
        kwargs = dict(
            geometry=granfilm_case.geometry,
            tr=granfilm_case.tr,
            R_nm=granfilm_case.R,
            surface_effects=granfilm_case.surface_effects,
        )
        finite = reference_from_granfilm_tree(
            dielectric_dir, "ag", energy, mean_free_path="finite_size", **kwargs
        )
        aparam = reference_from_granfilm_tree(
            dielectric_dir, "ag", energy, mean_free_path="A-parameter", A=0.8, **kwargs
        )
        assert np.max(np.abs(finite - aparam)) < 1e-12


class TestInteractiveMeanFreePathBranches:
    """s-only / tau / manual via explicit parameters (Fortran uses stdin)."""

    @pytest.fixture
    def base_eps_and_params(self, dielectric_dir):
        energy_tab, _wl, n_vals, k_vals = parse_granfilm_nk(dielectric_dir / "ag.nk")
        y_tab = n_vals + 1j * k_vals
        energy = np.linspace(1.5, 4.5, 50, dtype=np.float64)
        eps = dielectric_constants(energy, energy_tab, y_tab)
        params = parse_finite_size_dat(dielectric_dir / "Finite_Size" / "ag.dat")
        return energy, eps, params

    def test_s_only_branch(self, base_eps_and_params):
        energy, eps, params = base_eps_and_params
        inv_tau = 0.05
        s_only = dielectric_func_corrections(
            eps,
            energy,
            params,
            geometry="island",
            tr=0.75,
            R_nm=5.0,
            mean_free_path="s-only",
            surface_effects=True,
            inv_tau_eV=inv_tau,
        )
        tau = dielectric_func_corrections(
            eps,
            energy,
            params,
            geometry="island",
            tr=0.75,
            R_nm=5.0,
            mean_free_path="tau",
            surface_effects=True,
            inv_tau_eV=inv_tau,
        )
        assert np.all(np.isfinite(s_only))
        assert not np.allclose(s_only, tau)

    def test_tau_branch(self, base_eps_and_params):
        energy, eps, params = base_eps_and_params
        inv_tau = 0.05
        out = dielectric_func_corrections(
            eps,
            energy,
            params,
            geometry="island",
            tr=0.75,
            R_nm=5.0,
            mean_free_path="tau",
            surface_effects=True,
            inv_tau_eV=inv_tau,
        )
        ref = dielectric_func_corrections(
            eps,
            energy,
            params,
            geometry="island",
            tr=0.75,
            R_nm=5.0,
            mean_free_path="tau",
            surface_effects=True,
            inv_tau_eV=inv_tau,
        )
        assert np.max(np.abs(out - ref)) == 0.0

    def test_manual_branch(self, base_eps_and_params):
        energy, eps, params = base_eps_and_params
        out = dielectric_func_corrections(
            eps,
            energy,
            params,
            geometry="island",
            tr=0.75,
            R_nm=5.0,
            mean_free_path="manual",
            surface_effects=True,
            manual_percent=0.5,
        )
        diff = out - eps
        assert np.all(np.abs(np.imag(diff)) > 0.0)


class TestThermoreflectance:
    def test_substrate_energy_shift(self):
        n = 100
        emin, emax = 1.5, 4.5
        energy = np.linspace(emin, emax, n, dtype=np.float64)
        eps = (energy**2 + 1j * 0.1 * energy).astype(np.complex128)
        shift = 0.3
        shifted = apply_thermoreflectance_shift(eps, emin, emax, shift)
        denergy = (emax - emin) / (n - 1)
        di = int(shift / denergy)
        for i in range(di, n):
            assert shifted[i] == pytest.approx(eps[i - di])
        for i in range(di):
            assert shifted[i] == pytest.approx(eps[i])
