"""Load GranFilm Fortran baseline DR spectra."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from granfilm.common.constants import DEFAULT_GRANFILM_DIR


@dataclass(frozen=True)
class BaselineSpectrum:
    energy_ev: np.ndarray
    value: np.ndarray
    wavelength_nm: np.ndarray
    source: str


def granfilm_dir() -> Path:
    return Path(DEFAULT_GRANFILM_DIR).resolve()


def default_baseline_path() -> Path:
    env = os.environ.get("GRANFILM_BASELINE", "").strip()
    if env:
        return Path(env).resolve()
    return granfilm_dir() / "testing" / "SphereTest.dat"


def default_spheroid_baseline_path() -> Path:
    env = os.environ.get("GRANFILM_SPHEROID_BASELINE", "").strip()
    if env:
        return Path(env).resolve()
    return granfilm_dir() / "testing" / "SpheroidTest.dat"


def default_prolate_baseline_path() -> Path:
    env = os.environ.get("GRANFILM_PROLATE_BASELINE", "").strip()
    if env:
        return Path(env).resolve()
    return granfilm_dir() / "testing" / "ProlateTest.dat"


def default_yamaguchi_baseline_path() -> Path:
    env = os.environ.get("GRANFILM_YAMAGUCHI_BASELINE", "").strip()
    if env:
        return Path(env).resolve()
    return granfilm_dir() / "testing" / "YamaguchiTest.dat"


def default_coated_baseline_path() -> Path:
    env = os.environ.get("GRANFILM_COATED_BASELINE", "").strip()
    if env:
        return Path(env).resolve()
    return granfilm_dir() / "testing" / "CoatedTest.dat"


def default_film_baseline_path() -> Path:
    return granfilm_dir() / "testing" / "FilmTest.dat"


def default_2film_baseline_path() -> Path:
    return granfilm_dir() / "testing" / "Film2Test.dat"


def default_thin_cap_baseline_path() -> Path:
    return granfilm_dir() / "testing" / "ThinCapTest.dat"


def default_quadrupole_baseline_path() -> Path:
    return granfilm_dir() / "testing" / "QuadrupoleTest.dat"


def default_square_baseline_path() -> Path:
    return granfilm_dir() / "testing" / "SquareTest.dat"


def default_hexagonal_baseline_path() -> Path:
    return granfilm_dir() / "testing" / "HexagonalTest.dat"


def default_below_baseline_path() -> Path:
    return granfilm_dir() / "testing" / "BelowTest.dat"


def default_invariants_baseline_path() -> Path:
    return granfilm_dir() / "testing" / "InvariantsTest.dat"


def default_aspnes_baseline_path() -> Path:
    return granfilm_dir() / "testing" / "AspnesTest.dat"


GEOMETRY_BASELINE_PATHS: dict[str, Path] = {
    "film": default_film_baseline_path(),
    "2film": default_2film_baseline_path(),
    "thin_cap": default_thin_cap_baseline_path(),
    "quadrupole": default_quadrupole_baseline_path(),
    "square": default_square_baseline_path(),
    "hexagonal": default_hexagonal_baseline_path(),
    "island": default_baseline_path(),
    "island_below": default_below_baseline_path(),
    "invariants": default_invariants_baseline_path(),
    "aspnes": default_aspnes_baseline_path(),
}


def parse_sphere_test_dat(path: Path) -> BaselineSpectrum:
    """Parse GranFilm result .dat (SphereTest.dat / result.dat)."""
    energy: list[float] = []
    values: list[float] = []
    wl: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 3:
            continue
        try:
            energy.append(float(parts[0]))
            values.append(float(parts[1]))
            wl.append(float(parts[2]))
        except ValueError:
            continue
    if not energy:
        raise ValueError(f"No numeric rows in baseline file: {path}")
    return BaselineSpectrum(
        energy_ev=np.asarray(energy, dtype=np.float64),
        value=np.asarray(values, dtype=np.float64),
        wavelength_nm=np.asarray(wl, dtype=np.float64),
        source=str(path),
    )


parse_spheroid_test_dat = parse_sphere_test_dat


def load_baseline(path: Path | str | None = None) -> BaselineSpectrum:
    p = Path(path) if path is not None else default_baseline_path()
    if p.suffix == ".npz":
        data = np.load(p)
        return BaselineSpectrum(
            energy_ev=data["energy_ev"],
            value=data["value"],
            wavelength_nm=data["wavelength_nm"],
            source=str(p),
        )
    return parse_sphere_test_dat(p)


def save_baseline_npz(spec: BaselineSpectrum, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        energy_ev=spec.energy_ev,
        value=spec.value,
        wavelength_nm=spec.wavelength_nm,
        source=spec.source,
    )


def granfilm_sphere_binary() -> Path | None:
    exe = granfilm_dir() / "src" / "Sphere" / "Sphere"
    return exe if exe.is_file() and os.access(exe, os.X_OK) else None


def run_granfilm_sphere_noninteractive(
    cwd: Path | None = None,
    *,
    stdin: str = "n\n",
) -> subprocess.CompletedProcess[str]:
    """Run compiled Sphere in testing/; default stdin exits after one calculation."""
    root = granfilm_dir()
    testing = cwd or (root / "testing")
    exe = granfilm_sphere_binary()
    if exe is None:
        raise FileNotFoundError(
            f"GranFilm Sphere binary not found under {root / 'src' / 'Sphere'}. "
            "Run granfilm/run_granfilm_baseline.py --build first."
        )
    return subprocess.run(
        [str(exe)],
        cwd=str(testing),
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )


def regenerate_baseline_dat(
    output_npz: Path | None = None,
    *,
    outfilename: str = "result",
) -> BaselineSpectrum:
    """Run compiled Sphere in testing/ and parse {outfilename}.dat."""
    root = granfilm_dir()
    testing = root / "testing"
    result_dat = testing / f"{outfilename}.dat"
    if granfilm_sphere_binary() is not None:
        proc = run_granfilm_sphere_noninteractive(testing)
        if proc.returncode != 0:
            raise RuntimeError(
                f"GranFilm Sphere failed (code {proc.returncode}):\n{proc.stderr[-2000:]}"
            )
        if not result_dat.is_file():
            raise FileNotFoundError(f"Expected {result_dat} after GranFilm run")
        spec = parse_sphere_test_dat(result_dat)
    else:
        spec = parse_sphere_test_dat(testing / "SphereTest.dat")
    if output_npz is not None:
        save_baseline_npz(spec, output_npz)
    return spec


def run_sphere_with_inc(inc_path: Path, *, granfilm_root: Path | None = None) -> Path:
    """
    Copy inc_path to testing/Sphere.inc, run Fortran Sphere, restore Sphere.inc.
    Returns path to the generated .dat (from outfilename in the inc).
    """
    from granfilm.sphere_island.case import load_sphere_inc

    root = granfilm_root or granfilm_dir()
    testing = root / "testing"
    sphere_inc = testing / "Sphere.inc"
    backup = sphere_inc.read_text(encoding="utf-8") if sphere_inc.is_file() else None
    case = load_sphere_inc(inc_path)
    output_dat = testing / f"{case.outfilename}.dat"
    try:
        sphere_inc.write_text(inc_path.read_text(encoding="utf-8"), encoding="utf-8")
        geom = case.geometry.strip().lower()
        if geom == "2film":
            stdin = "go\nn\n"
        elif geom == "coated":
            # Coating_Yamaguchi calls dielectric_func_corrections on coating; missing
            # Finite_Size/<coating>.dat triggers Pause — resume with go.
            stdin = "go\nn\n"
        else:
            stdin = "n\n"
        proc = run_granfilm_sphere_noninteractive(testing, stdin=stdin)
        if not output_dat.is_file():
            if proc.returncode != 0:
                raise RuntimeError(
                    f"GranFilm Sphere failed (code {proc.returncode}):\n{proc.stderr[-2000:]}"
                )
            raise FileNotFoundError(f"Expected {output_dat} after GranFilm run")
        return output_dat
    finally:
        if backup is not None:
            sphere_inc.write_text(backup, encoding="utf-8")


def run_build_script(repo_root: Path | None = None) -> int:
    root = repo_root or Path(__file__).resolve().parents[2]
    script = root / "granfilm" / "sphere_island" / "run_granfilm_baseline.py"
    return subprocess.call([sys.executable, str(script), "--build"], cwd=str(root))


def granfilm_spheroid_binary() -> Path | None:
    exe = granfilm_dir() / "src" / "Spheroid" / "Spheroid"
    return exe if exe.is_file() and os.access(exe, os.X_OK) else None


def run_granfilm_spheroid_noninteractive(
    cwd: Path | None = None,
    *,
    stdin: str = "n\n",
) -> subprocess.CompletedProcess[str]:
    """Run compiled Spheroid in testing/; default stdin exits after one calculation."""
    root = granfilm_dir()
    testing = cwd or (root / "testing")
    exe = granfilm_spheroid_binary()
    if exe is None:
        raise FileNotFoundError(
            f"GranFilm Spheroid binary not found under {root / 'src' / 'Spheroid'}. "
            "Run granfilm/oblate_prolate/run_granfilm_baseline.py --build first."
        )
    return subprocess.run(
        [str(exe)],
        cwd=str(testing),
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )


def run_spheroid_with_inc(inc_path: Path, *, granfilm_root: Path | None = None) -> Path:
    """
    Copy inc_path to testing/Spheroid.inc, run Fortran Spheroid, restore Spheroid.inc.
    Returns path to the generated .dat (from outfilename in the inc).
    """
    from granfilm.oblate_prolate.case import load_spheroid_inc

    root = granfilm_root or granfilm_dir()
    testing = root / "testing"
    spheroid_inc = testing / "Spheroid.inc"
    backup = spheroid_inc.read_text(encoding="utf-8") if spheroid_inc.is_file() else None
    case = load_spheroid_inc(inc_path)
    output_dat = testing / f"{case.outfilename}.dat"
    try:
        spheroid_inc.write_text(inc_path.read_text(encoding="utf-8"), encoding="utf-8")
        geom = case.geometry.strip().lower()
        if geom == "coated":
            stdin = "go\nn\n"
        else:
            stdin = "n\n"
        proc = run_granfilm_spheroid_noninteractive(testing, stdin=stdin)
        if not output_dat.is_file():
            if proc.returncode != 0:
                raise RuntimeError(
                    f"GranFilm Spheroid failed (code {proc.returncode}):\n{proc.stderr[-2000:]}"
                )
            raise FileNotFoundError(f"Expected {output_dat} after GranFilm run")
        return output_dat
    finally:
        if backup is not None:
            spheroid_inc.write_text(backup, encoding="utf-8")
