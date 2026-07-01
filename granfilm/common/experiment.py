"""Load GranFilm experimental DR spectra and compute chi2 reliance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from granfilm.common.baseline import granfilm_dir

EXPERIMENT_HEADER_LINES = 5


@dataclass(frozen=True)
class ExperimentSpectrum:
    energy_ev: np.ndarray
    dr: np.ndarray
    source: str


def resolve_experiment_dat_path(
    path_dielectric: str,
    expfilename: str,
    *,
    granfilm_root: Path | None = None,
) -> Path:
    """Resolve {expfilename}.dat under testing/ (Fortran CWD; path_dielectric is relative there)."""
    root = granfilm_root or granfilm_dir()
    testing = root / "testing"
    _ = (testing / Path(path_dielectric.strip().strip("'"))).resolve()
    return testing / f"{expfilename.strip()}.dat"


def load_experiment_dat(path: Path | str) -> ExperimentSpectrum:
    """Parse GranFilm experiment .dat (write_mod.f90: skip 5 headers, read E and DR)."""
    p = Path(path)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) <= EXPERIMENT_HEADER_LINES:
        raise ValueError(f"Experiment file too short (need >{EXPERIMENT_HEADER_LINES} lines): {p}")

    energy: list[float] = []
    dr: list[float] = []
    for line in lines[EXPERIMENT_HEADER_LINES:]:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            energy.append(float(parts[0]))
            dr.append(float(parts[1]))
        except ValueError:
            continue

    if not energy:
        raise ValueError(f"No numeric E/DR rows in experiment file: {p}")

    return ExperimentSpectrum(
        energy_ev=np.asarray(energy, dtype=np.float64),
        dr=np.asarray(dr, dtype=np.float64),
        source=str(p),
    )


def _locate_fortran(xx: np.ndarray, x: float) -> int:
    """Match Sphere global_def.f90 locate() for linear interpolation bracket."""
    n = len(xx)
    ascnd = xx[n - 1] >= xx[0]
    jl = 0
    ju = n + 1
    while ju - jl > 1:
        jm = (ju + jl) // 2
        if ascnd == (x >= xx[jm - 1]):
            jl = jm
        else:
            ju = jm
    if x == xx[0]:
        return 1
    if x == xx[-1]:
        return n - 1
    return jl


def interpolate_experiment_to_energy(exp: ExperimentSpectrum, energy_ev: np.ndarray) -> np.ndarray:
    """Linearly interpolate experiment DR onto the theory energy grid (write_mod.f90)."""
    e_exp = exp.energy_ev
    dr_exp = exp.dr
    out = np.empty(len(energy_ev), dtype=np.float64)
    npts = len(e_exp)
    for i, e in enumerate(energy_ev):
        start = _locate_fortran(e_exp, float(e))
        if start <= 0 or start >= npts:
            raise ValueError(
                f"Energy {e} eV outside experiment range [{e_exp[0]}, {e_exp[-1]}] ({exp.source})"
            )
        slope = (dr_exp[start] - dr_exp[start - 1]) / (e_exp[start] - e_exp[start - 1])
        out[i] = dr_exp[start - 1] + slope * (e - e_exp[start - 1])
    return out


def chi2_reliance(theory: np.ndarray, experiment_on_grid: np.ndarray) -> float:
    """Reliance factor chi2 = sqrt(sum((theory-exp)^2) / N) (write_mod.f90)."""
    theory = np.asarray(theory, dtype=np.float64)
    experiment_on_grid = np.asarray(experiment_on_grid, dtype=np.float64)
    if theory.shape != experiment_on_grid.shape:
        raise ValueError(
            f"shape mismatch: theory {theory.shape} vs experiment {experiment_on_grid.shape}"
        )
    n = theory.size
    if n == 0:
        raise ValueError("empty arrays for chi2")
    return float(np.sqrt(np.sum((theory - experiment_on_grid) ** 2) / n))
