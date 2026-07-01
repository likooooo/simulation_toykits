"""Write GranFilm-compatible result .dat files (write_mod.f90)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from granfilm.common.constants import HBAR_EV_S, C_NM_S


def _wavelength_nm(energy_ev: float) -> float:
    return float(2.0 * np.pi * HBAR_EV_S * C_NM_S / energy_ev)


def write_fresnel_dat(
    path: Path | str,
    *,
    case: Any,
    energy_ev: np.ndarray,
    values: np.ndarray,
    density: float | None = None,
    mean_thickness: float | None = None,
    shape_ratio: float | None = None,
) -> None:
    """Write {outfilename}.dat with Fortran write_mod header and E20.8 data rows."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    energy_ev = np.asarray(energy_ev, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    geom = str(getattr(case, "geometry", "")).strip().lower()
    out_kind = str(getattr(case, "out", "DR")).strip().upper()

    i_max = int(np.argmax(values))
    i_min = int(np.argmin(values))

    lines: list[str] = ["# HEADER "]
    lines.append(f"# Geometry             : {getattr(case, 'geometry', '')}")
    lines.append(f"# Calculation of       : {getattr(case, 'out', 'DR')}")
    lines.append(f"# Energy min           : {getattr(case, 'energy_min', energy_ev[0])}")
    lines.append(f"# Energy max           : {getattr(case, 'energy_max', energy_ev[-1])}")
    lines.append(f"# Incident angle (rad) : {getattr(case, 'theta0', 0.0)}")
    lines.append(f"# Polarisation         : {getattr(case, 'polarization', 'p')}")
    lines.append(f"# Film material        : {getattr(case, 'island', '')}")
    lines.append(f"# Substrate material   : {getattr(case, 'substrate', '')}")
    lines.append(f"# Fresnel formulae     : {getattr(case, 'fresnel', '')}")
    lines.append(f"# Finite size effects  : {getattr(case, 'mean_free_path', '')}")
    lines.append(f"# Surface effects      : {getattr(case, 'surface_effects', False)}")
    lines.append(f"# Temperature (K)      : {getattr(case, 'temperature', 300.0)}")

    if geom in {"film", "2film"}:
        lines.append(f"# Thichkness1 (nm) : {getattr(case, 'film_thickness1', 0.0)}")
        lines.append(f"# Thichkness2 (nm) : {getattr(case, 'film_thickness2', 0.0)}")
        lines.append(f"# Coating material : {getattr(case, 'coating', '')}")
    else:
        lines.append(f"# Interactions         : {getattr(case, 'interaction', '')}")
        r_nm = getattr(case, "R", getattr(case, "R_par", None))
        if r_nm is not None:
            lines.append(f"# Island radius (nm)   : {r_nm}")
        tr = getattr(case, "tr", None)
        if tr is not None:
            lines.append(f"# Truncation parameter : {tr}")
        cov = getattr(case, "coverage", None)
        if cov is not None:
            lines.append(f"# Coverage             : {cov}")
        lines.append(f"# Geometry             : {getattr(case, 'geometry', '')}")
        lines.append(f"# Network              : {getattr(case, 'network', '')}")
        lat = getattr(case, "lattice_const", None)
        if lat is not None:
            lines.append(f"# Lattice constant (nm): {lat}")
        mppos = getattr(case, "MPpos", None)
        if mppos is not None:
            lines.append(f"# Multipole position   : {mppos}")
        mpo = getattr(case, "Mpole_order", None)
        if mpo is not None:
            lines.append(f"# # multipoles         : {mpo}")
        if shape_ratio is not None:
            lines.append(f"# Shape ratio          : {shape_ratio}")
        if density is not None:
            lines.append(f"# Density              : {density}")
        if mean_thickness is not None:
            lines.append(f"# Mean thichkness (nm) : {mean_thickness}")

    lines.append(f"# Max  at :   E = {energy_ev[i_max]:g}   value = {values[i_max]:g}")
    lines.append(f"# Min  at :   E = {energy_ev[i_min]:g}   value = {values[i_min]:g}")
    lines.append("#")
    lines.append("# FORMAT:  Energy (eV)         Value          Wavelength (nm)")
    lines.append("#")

    for e, v in zip(energy_ev, values, strict=True):
        wl = _wavelength_nm(float(e))
        lines.append(f"{e:20.8E}{v:20.8E}{wl:20.8E}")

    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
