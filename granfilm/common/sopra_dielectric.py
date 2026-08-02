"""Fortran-compatible SOPRA .nk dielectric pipeline (GranFilm initialize_mod)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

HC_UM_EV = 1.243  # initialize_mod.f90 case(2)


@dataclass(frozen=True)
class FiniteSizeParams:
    omega_p_eV: float
    inv_relaxation_time_eV: float
    fermi_velocity_eV_nm: float
    surf_eff_eV2: float
    disp_T_eV_per_K: float

    @classmethod
    def from_yaml(cls, path: Path) -> FiniteSizeParams:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(
            omega_p_eV=float(data["omega_p_eV"]),
            inv_relaxation_time_eV=float(data["inv_relaxation_time_eV"]),
            fermi_velocity_eV_nm=float(data["fermi_velocity_eV_nm"]),
            surf_eff_eV2=float(data["surf_eff_eV2"]),
            disp_T_eV_per_K=float(data["disp_T_eV_per_K"]),
        )


def parse_granfilm_nk(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (energy_ev, wl_um, n, k) matching Fortran dielectric_constants."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = lines[0].split()
    unit = int(float(header[0]))
    x1, x2, nlines = float(header[1]), float(header[2]), int(float(header[3]))
    dx = (x2 - x1) / (nlines - 1)
    data_lines = [ln for ln in lines[1:] if ln.strip()]

    n_vals = np.empty(nlines, dtype=float)
    k_vals = np.empty(nlines, dtype=float)
    energy = np.empty(nlines, dtype=float)

    if unit == 1:
        for i in range(nlines):
            parts = data_lines[i].split()
            n_vals[i] = float(parts[0])
            k_vals[i] = float(parts[1])
            energy[i] = x1 + i * dx
    elif unit == 2:
        for i in range(nlines):
            row = nlines - 1 - i
            parts = data_lines[row].split()
            n_vals[i] = float(parts[0])
            k_vals[i] = float(parts[1])
            wl_raw = x1 + (nlines - 1 - i) * dx
            energy[i] = HC_UM_EV / wl_raw
    else:
        raise ValueError(f"unsupported .nk unit {unit} in {path}")

    wl_um = HC_UM_EV / energy
    return energy, wl_um, n_vals, k_vals


def parse_finite_size_dat(path: Path) -> FiniteSizeParams:
    values: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.split("!")[0].strip()
        if not stripped:
            continue
        values.append(float(stripped.split()[0]))
        if len(values) == 5:
            break
    if len(values) != 5:
        raise ValueError(f"expected 5 values in {path}, got {len(values)}")
    return FiniteSizeParams(
        omega_p_eV=values[0],
        inv_relaxation_time_eV=values[1],
        fermi_velocity_eV_nm=values[2],
        surf_eff_eV2=values[3],
        disp_T_eV_per_K=values[4],
    )


def locate(xx: np.ndarray, x: float) -> int:
    """NR locate() from GranFilm global_def.f90 (0-based; Fortran uses real(sp))."""
    xx = np.asarray(xx, dtype=np.float32)
    x32 = np.float32(x)
    n = xx.size
    ascnd = xx[-1] >= xx[0]
    jl, ju = 0, n + 1
    while ju - jl > 1:
        jm = (ju + jl) // 2
        if ascnd == (x32 >= xx[jm - 1]):
            jl = jm
        else:
            ju = jm
    if x32 == xx[0]:
        return 0
    if x32 == xx[-1]:
        return n - 2
    return jl - 1


def tabulated_y_from_material(mat: object) -> tuple[np.ndarray, np.ndarray]:
    """Recover uniform eV grid and y=n+ik from gf tabulated nk (hc=1.243 µm·eV)."""
    wl_um, n_vals, k_vals = mat.get_tabulated_values()
    wl = np.asarray(wl_um, dtype=np.float64)
    n_arr = np.asarray(n_vals, dtype=np.float64)
    k_arr = np.asarray(k_vals, dtype=np.float64)
    energy = HC_UM_EV / wl
    order = np.argsort(energy)
    energy = energy[order]
    y = (n_arr[order] + 1j * k_arr[order]).astype(np.complex128)
    return energy, y


def dielectric_constants(energy: np.ndarray, energy_tab: np.ndarray, y_tab: np.ndarray) -> np.ndarray:
    """Linear interp of n+ik on eV, then square (Fortran dielectric_constants)."""
    energy = np.asarray(energy, dtype=np.float64)
    eps = np.empty(len(energy), dtype=np.complex128)
    ntab = energy_tab.size
    for i, e in enumerate(energy):
        start = locate(energy_tab, float(e))
        if start < 0 or start >= ntab - 1:
            raise ValueError(f"energy {e} outside tabulated range [{energy_tab[0]}, {energy_tab[-1]}]")
        slope = (y_tab[start + 1] - y_tab[start]) / (energy_tab[start + 1] - energy_tab[start])
        y = y_tab[start] + slope * (e - energy_tab[start])
        eps[i] = y * y
    return eps


def apply_thermoreflectance_shift(
    eps_substrate: np.ndarray,
    energy_min: float,
    energy_max: float,
    energy_shift_ev: float,
) -> np.ndarray:
    """Shift substrate ε toward higher energy (initialize_dielectric_constants ~376-384)."""
    out = np.asarray(eps_substrate, dtype=np.complex128).copy()
    n = out.size
    if n < 2:
        return out
    denergy = (energy_max - energy_min) / (n - 1)
    di = int(energy_shift_ev / denergy)
    for i in range(n - 1, di - 1, -1):
        out[i] = eps_substrate[i - di]
    return out


def dielectric_func_corrections(
    epsilon: np.ndarray,
    energy: np.ndarray,
    params: FiniteSizeParams,
    *,
    geometry: str,
    tr: float,
    R_nm: float,
    mean_free_path: str,
    surface_effects: bool,
    temperature_k: float = 300.0,
    A: float = 0.8,
    R_par_nm: float | None = None,
    R_eff_nm: float | None = None,
    inv_tau_eV: float | None = None,
    manual_percent: float | None = None,
) -> np.ndarray:
    """Port of initialize_mod.f90 dielectric_func_corrections."""
    eps = np.asarray(epsilon, dtype=np.complex128).copy()
    energy = np.asarray(energy, dtype=np.float64)

    omega_p = params.omega_p_eV + params.disp_T_eV_per_K * (temperature_k - 300.0)
    inv_tau_bulk = params.inv_relaxation_time_eV
    fermi_velocity = params.fermi_velocity_eV_nm
    surf_eff = params.surf_eff_eV2

    mfp_raw = mean_free_path.strip()
    mfp = mfp_raw.lower()
    if mfp == "none":
        return eps

    geom = geometry.strip().lower()

    # Fortran initialize_mod.f90 select case only sets R_eff for film/2film/island.
    # For thin_cap, R_eff is left at 0 → inv_tau = Inf → tmp_surf_eff = 0: only bulk Drude add.
    if geom == "thin_cap":
        if mfp != "finite_size":
            raise ValueError(f"thin_cap mean_free_path={mean_free_path!r}")
        for i, e in enumerate(energy):
            eps[i] = eps[i] + omega_p**2 / (e**2 + 1j * e * inv_tau_bulk)
        return eps

    if geom in {"film", "2film"}:
        r_eff = R_nm
    elif geom == "island":
        r_eff = (1.0 + tr) * R_nm
        if R_par_nm is not None and r_eff > R_par_nm:
            r_eff = R_par_nm
    else:
        r_eff = (1.0 + tr) * R_nm

    if R_eff_nm is not None:
        r_eff = R_eff_nm

    if mfp in {"finite_size", "a-parameter"}:
        inv_tau = inv_tau_bulk + A * fermi_velocity / r_eff
    elif mfp in {"s-only", "tau"}:
        if inv_tau_eV is None:
            raise ValueError(f"mean_free_path={mfp_raw!r} requires inv_tau_eV")
        inv_tau = inv_tau_eV
    elif mfp == "manual":
        if manual_percent is None:
            raise ValueError("mean_free_path='manual' requires manual_percent")
        inv_tau = inv_tau_bulk + A * fermi_velocity / r_eff
    else:
        raise ValueError(f"mean_free_path={mean_free_path!r}")

    for i, e in enumerate(energy):
        if surface_effects:
            tmp_surf_eff = omega_p**2 / (e**2 + surf_eff / r_eff + 1j * e * inv_tau)
        else:
            tmp_surf_eff = omega_p**2 / (e**2 + 1j * e * inv_tau)
        eps[i] = eps[i] + omega_p**2 / (e**2 + 1j * e * inv_tau_bulk)
        if mfp == "s-only":
            eps[i] = np.real(eps[i]) - tmp_surf_eff
        elif mfp == "tau":
            eps[i] = eps[i] - tmp_surf_eff
        elif mfp == "manual":
            diff = eps[i] - tmp_surf_eff
            eps[i] = np.real(diff) + 1j * manual_percent * np.imag(diff)
        elif mfp in {"finite_size", "a-parameter"}:
            eps[i] = eps[i] - tmp_surf_eff
        else:
            raise ValueError(f"mean_free_path branch {mfp_raw!r}")
    return eps


def epsilon_from_gf_material(
    mat: object,
    energy_ev: np.ndarray,
    *,
    finite_size: FiniteSizeParams | None = None,
    geometry: str = "island",
    tr: float = 0.0,
    R_nm: float = 5.0,
    R_par_nm: float | None = None,
    mean_free_path: str = "none",
    surface_effects: bool = False,
    temperature_k: float = 300.0,
    A: float = 0.8,
    inv_tau_eV: float | None = None,
    manual_percent: float | None = None,
) -> np.ndarray:
    energy_tab, y_tab = tabulated_y_from_material(mat)
    eps = dielectric_constants(energy_ev, energy_tab, y_tab)
    if finite_size is not None and mean_free_path.strip().lower() != "none":
        eps = dielectric_func_corrections(
            eps,
            energy_ev,
            finite_size,
            geometry=geometry,
            tr=tr,
            R_nm=R_nm,
            R_par_nm=R_par_nm,
            mean_free_path=mean_free_path,
            surface_effects=surface_effects,
            temperature_k=temperature_k,
            A=A,
            inv_tau_eV=inv_tau_eV,
            manual_percent=manual_percent,
        )
    return eps


def reference_from_granfilm_tree(
    dielectric_dir: Path,
    material: str,
    energy_ev: np.ndarray,
    *,
    geometry: str = "island",
    tr: float = 0.0,
    R_nm: float = 5.0,
    mean_free_path: str = "finite_size",
    surface_effects: bool = True,
    temperature_k: float = 300.0,
    A: float = 0.8,
    inv_tau_eV: float | None = None,
    manual_percent: float | None = None,
) -> np.ndarray:
    """Reference ε(ω) from GranFilm Dielectric/*.nk (+ Finite_Size/*.dat if present)."""
    nk_path = dielectric_dir / f"{material}.nk"
    energy_tab, _wl, n_vals, k_vals = parse_granfilm_nk(nk_path)
    y_tab = n_vals + 1j * k_vals
    eps = dielectric_constants(energy_ev, energy_tab, y_tab)
    fs_path = dielectric_dir / "Finite_Size" / f"{material}.dat"
    if fs_path.is_file() and mean_free_path.strip().lower() != "none":
        params = parse_finite_size_dat(fs_path)
        eps = dielectric_func_corrections(
            eps,
            energy_ev,
            params,
            geometry=geometry,
            tr=tr,
            R_nm=R_nm,
            mean_free_path=mean_free_path,
            surface_effects=surface_effects,
            temperature_k=temperature_k,
            A=A,
            inv_tau_eV=inv_tau_eV,
            manual_percent=manual_percent,
        )
    return eps
