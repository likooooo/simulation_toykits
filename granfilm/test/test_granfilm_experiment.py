"""GranFilm experimental spectrum loading and chi2 reliance."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from granfilm.common.baseline import granfilm_dir, parse_sphere_test_dat
from granfilm.common.write_output import write_fresnel_dat
from granfilm.sphere_island.case import default_sphere_case
from granfilm.common.experiment import (
    chi2_reliance,
    interpolate_experiment_to_energy,
    load_experiment_dat,
    resolve_experiment_dat_path,
)


def _write_minimal_experiment_dat(path: Path, energy: np.ndarray, dr: np.ndarray) -> None:
    header = "\n".join(f"# header {i}" for i in range(5))
    rows = "\n".join(f"{e:.6f}  {v:.6f}" for e, v in zip(energy, dr, strict=True))
    path.write_text(f"{header}\n{rows}\n", encoding="utf-8")


@pytest.fixture
def sample_experiment_dat(tmp_path: Path) -> Path:
    energy = np.linspace(1.5, 4.5, 64)
    dr = 0.1 + 0.05 * np.sin(energy)
    path = tmp_path / "agmgo.dat"
    _write_minimal_experiment_dat(path, energy, dr)
    return path


class TestLoadExperiment:
    def test_load_skips_five_header_lines(self, sample_experiment_dat: Path) -> None:
        spec = load_experiment_dat(sample_experiment_dat)
        assert spec.energy_ev.shape == spec.dr.shape
        assert len(spec.energy_ev) == 64
        assert spec.energy_ev[0] == pytest.approx(1.5)
        assert spec.source.endswith("agmgo.dat")

    def test_granfilm_agmgo_if_present(self) -> None:
        path = granfilm_dir() / "testing" / "agmgo.dat"
        if not path.is_file():
            pytest.skip(f"GranFilm experiment fixture missing: {path}")
        spec = load_experiment_dat(path)
        assert len(spec.energy_ev) > 0
        assert np.all(np.isfinite(spec.dr))


class TestInterpolation:
    def test_interpolate_matches_np_on_interior(self, sample_experiment_dat: Path) -> None:
        spec = load_experiment_dat(sample_experiment_dat)
        grid = np.linspace(2.0, 4.0, 10)
        py = interpolate_experiment_to_energy(spec, grid)
        ref = np.interp(grid, spec.energy_ev, spec.dr)
        assert py == pytest.approx(ref, rel=0, abs=1e-12)

    def test_interpolate_out_of_range_raises(self, sample_experiment_dat: Path) -> None:
        spec = load_experiment_dat(sample_experiment_dat)
        with pytest.raises(ValueError, match="outside experiment range"):
            interpolate_experiment_to_energy(spec, np.array([1.0]))


class TestChi2Reliance:
    def test_identical_spectra_zero(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        assert chi2_reliance(x, x) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        theory = np.array([1.0, 2.0, 3.0])
        exp = np.array([1.1, 2.1, 2.9])
        assert chi2_reliance(theory, exp) == pytest.approx(0.1)


class TestResolvePath:
    def test_resolve_from_dielectric_inc_layout(self) -> None:
        root = Path(os.environ.get("GRANFILM_DIR", "/home/like/repos/GranFilm-v1.0"))
        path = resolve_experiment_dat_path("../Dielectric", "agmgo", granfilm_root=root)
        assert path == (root / "testing" / "agmgo.dat").resolve()


class TestWriteFresnelDat:
    def test_roundtrip_parse(self, tmp_path: Path) -> None:
        case = default_sphere_case()
        energy = np.linspace(1.5, 4.5, 5)
        values = 0.01 * np.sin(energy)
        out = tmp_path / "out.dat"
        write_fresnel_dat(out, case=case, energy_ev=energy, values=values)
        spec = parse_sphere_test_dat(out)
        assert spec.energy_ev == pytest.approx(energy)
        assert spec.value == pytest.approx(values)
        text = out.read_text(encoding="utf-8")
        assert "# HEADER" in text
        assert "# FORMAT:" in text
