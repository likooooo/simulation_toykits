"""Material nk alignment between OpenFilters abeles and simulation_database of/*."""

from __future__ import annotations

import sys
from typing import Sequence

from abeles_loader import load_abeles
from bootstrap_simulation import bootstrap_toykits_session
from materials_db import OF_ALIGNMENT_MATERIALS
from paths import database_of_dir, upstream_openfilters_root


def _of_parser_module():
    of_db = database_of_dir()
    if str(of_db) not in sys.path:
        sys.path.insert(0, str(of_db))
    import of_mat_parser

    return of_mat_parser


def openfilters_nk_at_wl(material_name: str, wl_nm: float) -> complex:
    parser = _of_parser_module()
    of_root = upstream_openfilters_root()
    mat_file = of_root / "materials" / f"{material_name}.mat"
    if not mat_file.is_file():
        raise FileNotFoundError(mat_file)
    mat = parser.parse_mat_file(mat_file)
    wl_um, n_vals, k_vals = parser.sample_nk_grid(
        mat, openfilters_dir=of_root, wl_nm_min=wl_nm, wl_nm_max=wl_nm, wl_nm_step=1.0
    )
    return complex(float(n_vals[0]), -float(k_vals[0]))


def simulation_nk_at_wl(token: str, wl_nm: float, sim_db) -> complex:
    path = OF_ALIGNMENT_MATERIALS[token]
    mat = sim_db.read_at_path(path)
    return complex(mat.nk_at_wavelength_um(wl_nm / 1000.0))


def compare_material_alignment(
    wls_nm: Sequence[float] | None = None,
    *,
    n_tol: float = 1e-4,
    k_tol: float = 1e-6,
) -> list[str]:
    bootstrap_toykits_session()
    from simulation_database_parser import get_simulation_database

    sim_db = get_simulation_database(init=True)
    wls = list(wls_nm or [400.0, 550.0, 700.0, 900.0])
    failures: list[str] = []
    token_to_mat = {
        "of_void": "void",
        "of_TiO2": "TiO2",
        "of_SiO2": "SiO2",
        "of_BK7": "BK7",
        "of_FusedSilica": "FusedSilica",
    }
    for token, mat_name in token_to_mat.items():
        for wl in wls:
            nk_of = openfilters_nk_at_wl(mat_name, wl)
            nk_sim = simulation_nk_at_wl(token, wl, sim_db)
            dn = abs(nk_of.real - nk_sim.real)
            dk = abs(abs(nk_of.imag) - abs(nk_sim.imag))
            if dn > n_tol or dk > k_tol:
                failures.append(
                    f"{token} @ {wl} nm: OF={nk_of} sim={nk_sim} |dn|={dn:.2e} |dk|={dk:.2e}"
                )
    return failures
