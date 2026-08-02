"""Parse GranFilm Sphere.inc into a Python case object."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GranFilmCase:
    """Parameters for Sphere / island geometry (testing/Sphere.inc)."""

    path_dielectric: str
    geometry: str
    film_thickness1: float
    film_thickness2: float
    R: float
    tr: float
    MPpos: float
    interaction: str
    network: str
    lattice_const: float
    coverage: float
    out: str
    polarization: str
    theta0: float
    fresnel: str
    energy_min: float
    energy_max: float
    Nenergy: int
    island: str
    substrate: str
    coating: str
    mean_free_path: str
    surface_effects: bool
    polarizability: bool
    normalization: bool
    strenght: bool
    bound: str
    BC_energy: float
    scaling: str
    Mpole_order: int
    outfilename: str
    compa: bool
    expfilename: str
    # Numerics (Fortran defaults when not in .inc)
    Nint: int = 250
    int_method: str = "gauleg"
    Levels: int = 500
    epslin: float = 1e-4
    temperature: float = 300.0

    @property
    def island_material_key(self) -> tuple[str, ...]:
        return ("gf", "materials", f"{self.island}.yml")

    @property
    def substrate_material_key(self) -> tuple[str, ...]:
        return ("gf", "materials", f"{self.substrate}.yml")


def _parse_bool(token: str) -> bool:
    return token.strip().upper() in {"T", "TRUE", ".TRUE."}


def load_sphere_inc(path: Path | str) -> GranFilmCase:
    """Read a Sphere.inc file (Fortran read_param layout)."""
    lines = [
        ln.strip()
        for ln in Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        if ln.strip() and not ln.strip().startswith("!")
    ]
    if len(lines) < 14:
        raise ValueError(f"Sphere.inc too short: {path}")

    p = lines[0].strip().strip("'").split("!")[0].strip()
    geom = lines[1].split()[0]
    t1, t2 = map(float, lines[2].split()[:2])
    R, tr, MPpos = map(float, lines[3].split()[:3])
    inter_parts = lines[4].split()
    interaction, network = inter_parts[0], inter_parts[1]
    lattice_const, coverage = float(inter_parts[2]), float(inter_parts[3])
    out_parts = lines[5].split()
    out_kind, pol, theta0, fresnel = out_parts[0], out_parts[1], float(out_parts[2]), out_parts[3]
    eparts = lines[6].split()
    emin, emax, nenergy = float(eparts[0]), float(eparts[1]), int(float(eparts[2]))
    mats = lines[7].split()
    island, substrate, coating = mats[0], mats[1], mats[2]
    mfp_parts = lines[8].split()
    mean_free_path = mfp_parts[0]
    surface_effects = _parse_bool(mfp_parts[1])
    flags = lines[9].split()
    polarizability, normalization, strenght = map(_parse_bool, flags[:3])
    bound_parts = lines[10].split()
    bound, bc_energy, scaling = bound_parts[0], float(bound_parts[1]), bound_parts[2]
    mpo = int(lines[11].split()[0])
    outfilename = lines[12].split()[0]
    compa_parts = lines[13].split()
    compa = _parse_bool(compa_parts[0])
    expfilename = compa_parts[1] if len(compa_parts) > 1 else ""

    return GranFilmCase(
        path_dielectric=p,
        geometry=geom,
        film_thickness1=t1,
        film_thickness2=t2,
        R=R,
        tr=tr,
        MPpos=MPpos,
        interaction=interaction,
        network=network,
        lattice_const=lattice_const,
        coverage=coverage,
        out=out_kind,
        polarization=pol,
        theta0=theta0,
        fresnel=fresnel,
        energy_min=emin,
        energy_max=emax,
        Nenergy=nenergy,
        island=island,
        substrate=substrate,
        coating=coating,
        mean_free_path=mean_free_path,
        surface_effects=surface_effects,
        polarizability=polarizability,
        normalization=normalization,
        strenght=strenght,
        bound=bound,
        BC_energy=bc_energy,
        scaling=scaling,
        Mpole_order=mpo,
        outfilename=outfilename,
        compa=compa,
        expfilename=expfilename,
    )


_INC_DIR = Path(__file__).resolve().parent / "inc"


def default_sphere_case(*, granfilm_dir: str | None = None) -> GranFilmCase:
    del granfilm_dir
    return load_sphere_inc(_INC_DIR / "SphereIsland.inc")


def default_island_below_case() -> GranFilmCase:
    return load_sphere_inc(_INC_DIR / "SphereIslandBelow.inc")


def default_film_case() -> GranFilmCase:
    return load_sphere_inc(_INC_DIR / "SphereFilm.inc")


def default_2film_case() -> GranFilmCase:
    return load_sphere_inc(_INC_DIR / "Sphere2Film.inc")


def default_thin_cap_case() -> GranFilmCase:
    return load_sphere_inc(_INC_DIR / "SphereThinCap.inc")


def default_quadrupole_case() -> GranFilmCase:
    return load_sphere_inc(_INC_DIR / "SphereQuadrupole.inc")


def default_square_case() -> GranFilmCase:
    return load_sphere_inc(_INC_DIR / "SphereSquare.inc")


def default_hexagonal_case() -> GranFilmCase:
    return load_sphere_inc(_INC_DIR / "SphereHexagonal.inc")


def default_invariants_case() -> GranFilmCase:
    return load_sphere_inc(_INC_DIR / "SphereInvariants.inc")


def default_aspnes_case() -> GranFilmCase:
    return load_sphere_inc(_INC_DIR / "SphereAspnes.inc")


def inc_path_for_geometry(geometry: str) -> Path:
    """Canonical inc template under granfilm/sphere_island/inc/."""
    mapping = {
        "film": "SphereFilm.inc",
        "2film": "Sphere2Film.inc",
        "thin_cap": "SphereThinCap.inc",
        "quadrupole": "SphereQuadrupole.inc",
        "square": "SphereSquare.inc",
        "hexagonal": "SphereHexagonal.inc",
        "island": "SphereIsland.inc",
        "island_below": "SphereIslandBelow.inc",
        "invariants": "SphereInvariants.inc",
        "aspnes": "SphereAspnes.inc",
    }
    key = geometry.strip().lower()
    if key not in mapping:
        raise ValueError(f"unknown geometry {geometry!r}")
    return _INC_DIR / mapping[key]
