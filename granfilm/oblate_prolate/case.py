"""Parse GranFilm Spheroid.inc into a Python case object."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpheroidCase:
    """Parameters for oblate/prolate island geometry (testing/Spheroid.inc)."""

    path_dielectric: str
    polarization: str
    energy_min: float
    energy_max: float
    theta0: float
    geometry: str
    island: str
    substrate: str
    coating: str
    mean_free_path: str
    surface_effects: bool
    R_par: float
    R_per: float
    thickness: float
    tr: float
    network: str
    coverage: float
    lattice_const: float
    interaction: str
    fresnel: str
    out: str
    strenght: bool
    polarizability: bool
    Nenergy: int
    Mpole_order: int
    bound: str
    BC_energy: float
    outfilename: str
    int_method: str
    compa: bool
    expfilename: str
    # Numerics (Fortran defaults when not in .inc)
    Nint: int = 250
    Levels: int = 500
    epslin: float = 1e-5
    temperature: float = 300.0

    @property
    def island_material_key(self) -> tuple[str, ...]:
        return ("gf", "materials", f"{self.island}.yml")

    @property
    def substrate_material_key(self) -> tuple[str, ...]:
        return ("gf", "materials", f"{self.substrate}.yml")


def _parse_bool(token: str) -> bool:
    return token.strip().upper() in {"T", "TRUE", ".TRUE."}


def _first_token(line: str) -> str:
    tok = line.strip().split("!")[0].strip().split()[0]
    return tok.strip("'").strip('"')


def load_spheroid_inc(path: Path | str) -> SpheroidCase:
    """Read a Spheroid.inc file (Fortran read_param layout)."""
    lines = [
        ln.strip()
        for ln in Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        if ln.strip() and not ln.strip().startswith("!")
    ]
    if len(lines) < 25:
        raise ValueError(f"Spheroid.inc too short: {path}")

    p = _first_token(lines[0])
    pol = _first_token(lines[1])
    emin, emax = float(_first_token(lines[2])), float(_first_token(lines[3]))
    theta0 = float(_first_token(lines[4]))
    geom = _first_token(lines[5])
    island, substrate, coating = (_first_token(lines[i]) for i in (6, 7, 8))
    mean_free_path = _first_token(lines[9])
    surface_effects = _parse_bool(_first_token(lines[10]))
    r_par = float(_first_token(lines[11]))
    r_per = float(_first_token(lines[12]))
    thickness = float(_first_token(lines[13]))
    tr = float(_first_token(lines[14]))
    network = _first_token(lines[15])
    coverage = float(_first_token(lines[16]))
    lattice_const = float(_first_token(lines[17]))
    interaction = _first_token(lines[18])
    fresnel = _first_token(lines[19])
    out_kind = _first_token(lines[20])
    strenght = _parse_bool(_first_token(lines[21]))
    polarizability = _parse_bool(_first_token(lines[22]))
    nenergy = int(float(_first_token(lines[23])))
    mpo = int(_first_token(lines[24]))
    bound = _first_token(lines[25])
    bc_energy = float(_first_token(lines[26]))
    outfilename = _first_token(lines[27])
    int_method = _first_token(lines[28])
    compa_parts = lines[29].split("!")[0].split()
    compa = _parse_bool(compa_parts[0])
    expfilename = compa_parts[1] if len(compa_parts) > 1 else ""

    return SpheroidCase(
        path_dielectric=p,
        polarization=pol,
        energy_min=emin,
        energy_max=emax,
        theta0=theta0,
        geometry=geom,
        island=island,
        substrate=substrate,
        coating=coating,
        mean_free_path=mean_free_path,
        surface_effects=surface_effects,
        R_par=r_par,
        R_per=r_per,
        thickness=thickness,
        tr=tr,
        network=network,
        coverage=coverage,
        lattice_const=lattice_const,
        interaction=interaction,
        fresnel=fresnel,
        out=out_kind,
        strenght=strenght,
        polarizability=polarizability,
        Nenergy=nenergy,
        Mpole_order=mpo,
        bound=bound,
        BC_energy=bc_energy,
        outfilename=outfilename,
        int_method=int_method,
        compa=compa,
        expfilename=expfilename,
    )


_PACKAGE_DIR = Path(__file__).resolve().parent


def spheroid_inc_path() -> Path:
    return _PACKAGE_DIR / "inc" / "Spheroid.inc"


def prolate_inc_path() -> Path:
    return _PACKAGE_DIR / "inc" / "SpheroidProlate.inc"


def yamaguchi_inc_path() -> Path:
    return _PACKAGE_DIR / "inc" / "SpheroidYamaguchi.inc"


def coated_inc_path() -> Path:
    return _PACKAGE_DIR / "inc" / "SpheroidCoated.inc"


def default_spheroid_case(*, granfilm_dir: str | None = None) -> SpheroidCase:
    del granfilm_dir
    return load_spheroid_inc(spheroid_inc_path())


def default_prolate_case(*, granfilm_dir: str | None = None) -> SpheroidCase:
    return load_spheroid_inc(prolate_inc_path())


def default_yamaguchi_case(*, granfilm_dir: str | None = None) -> SpheroidCase:
    return load_spheroid_inc(yamaguchi_inc_path())


def default_coated_case(*, granfilm_dir: str | None = None) -> SpheroidCase:
    return load_spheroid_inc(coated_inc_path())
