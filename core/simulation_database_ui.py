"""Simulation database helpers and Streamlit browser (logic from simulation_database_tui)."""

from __future__ import annotations

import glob
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core import simulation_loader


@dataclass
class NavFrame:
    sim_db: Any
    db_name: str | None = None
    oghma: Any = None
    query: Any = None
    path_keys: list[str] = field(default_factory=list)
    selected: int = 0

    @property
    def keys(self) -> list[str]:
        if self.db_name is None:
            return list(self.sim_db.database_names())
        return list(self.query.keys)

    def breadcrumb(self) -> str:
        if self.db_name is None:
            return "simulation_database"
        parts = [self.db_name, *self.path_keys]
        return " > ".join(parts)


def ensure_simulation_database_initialized(sim_db=None):
    simulation_loader.ensure_artifacts_on_path()
    if sim_db is not None:
        sim_db.init()
        return sim_db
    import simulation_database_parser as sdp

    return sdp.get_simulation_database(init=True)


def prepare_simulation_database(
    sim_db=None,
    db_name: str = "materials",
) -> tuple[Any, list[str]]:
    """Ensure database is installed via simulation_database_parser.get_simulation_database."""
    lines: list[str] = []
    try:
        sim_db = ensure_simulation_database_initialized(sim_db)
        lines.append(f"database root: {sim_db.root_path()}")
    except Exception as exc:
        lines.append(f"prepare failed: {exc}")
        return sim_db, lines

    if db_name not in list(sim_db.database_names()):
        lines.append(f"[{db_name}] unknown database")
        return sim_db, lines

    oghma = sim_db.database(db_name)
    lines.append(f"[{db_name}] local path: {oghma.local_path()}")
    lines.append(f"[{db_name}] ready")
    return sim_db, lines


def vacuum_material():
    sim = simulation_loader.get_simulation_module()
    return sim.material_s.from_nk(1.0 + 0.0j, "Vacuum")


def read_leaf_material(oghma: Any, leaf_query: Any) -> Any:
    mat = oghma.read(leaf_query)
    if mat is None:
        raise ValueError("read requires a leaf query_object")
    return mat


def _cplx(z) -> complex:
    if callable(getattr(z, "real", None)):
        return complex(z.real(), z.imag())
    return complex(z.real, z.imag)


def material_nk_arrays(mat: Any, num_samples: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (wl_um, n, k) arrays for plotting or CSV export."""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            mat.dump(tmp)
            wl_files = glob.glob(os.path.join(tmp, "*_wl_um.csv"))
            if wl_files:
                base = wl_files[0][:-len("_wl_um.csv")]
                wl = np.loadtxt(f"{base}_wl_um.csv")
                n = np.loadtxt(f"{base}_n.csv")
                k_path = f"{base}_k.csv"
                k = np.loadtxt(k_path) if os.path.isfile(k_path) else np.zeros_like(n)
                return np.asarray(wl), np.asarray(n), np.asarray(k)
    except Exception:
        pass

    wl_min, wl_max = 0.2, 2.5
    try:
        for probe in (0.4, 0.532, 0.6, 1.0, 1.55):
            _cplx(mat.nk_at_wavelength_um(probe))
        wl_min, wl_max = 0.2, 2.5
    except Exception:
        wl_min, wl_max = 0.4, 0.8
    wls = np.linspace(wl_min, wl_max, num_samples)
    nk = [_cplx(mat.nk_at_wavelength_um(w)) for w in wls]
    return wls, np.real(nk), np.imag(nk)


def material_to_csv_bytes(mat: Any) -> bytes:
    wl, n, k = material_nk_arrays(mat)
    import io

    buf = io.StringIO()
    buf.write("Wavelength(um),n,k\n")
    for w, nv, kv in zip(wl, n, k):
        buf.write(f"{w},{nv},{kv}\n")
    return buf.getvalue().encode("utf-8")


def new_nav_stack(sim_db=None) -> list[NavFrame]:
    if sim_db is None:
        sim_db = ensure_simulation_database_initialized()
    return [NavFrame(sim_db=sim_db)]


def descend_frame(stack: list[NavFrame], key: str) -> tuple[bool, str]:
    """Enter key at current frame. Returns (is_leaf, status_message)."""
    frame = stack[-1]
    if frame.db_name is None:
        oghma = frame.sim_db.database(key)
        stack.append(
            NavFrame(sim_db=frame.sim_db, db_name=key, oghma=oghma, query=oghma.query(), selected=0)
        )
        return False, f"进入 {key}"

    assert frame.oghma is not None
    if frame.query.is_leaf:
        return True, ""

    try:
        is_leaf, child = frame.query.descend(key)
    except Exception as exc:
        return False, f"无法进入 {key}: {exc}"

    if is_leaf:
        stack.append(
            NavFrame(
                sim_db=frame.sim_db,
                db_name=frame.db_name,
                oghma=frame.oghma,
                query=child,
                path_keys=[*frame.path_keys, key],
                selected=0,
            )
        )
        return True, f"已选中: {key}"

    stack.append(
        NavFrame(
            sim_db=frame.sim_db,
            db_name=frame.db_name,
            oghma=frame.oghma,
            query=child,
            path_keys=[*frame.path_keys, key],
            selected=0,
        )
    )
    return False, f"进入 {key}"


def current_leaf_query(stack: list[NavFrame]) -> Any | None:
    frame = stack[-1]
    if frame.db_name is None or frame.oghma is None:
        return None
    if frame.query.is_leaf:
        return frame.query
    return None


def search_material_paths(sim_db: Any, material_name: str, db_name: str = "materials") -> list[list[str]]:
    """Find leaf path key lists whose breadcrumb matches material_name (case-insensitive)."""
    key = (material_name or "").strip()
    if not key:
        return []
    oghma = sim_db.database(db_name)
    matches: list[list[str]] = []
    pattern = re.compile(re.escape(key), re.IGNORECASE)

    def walk(query, prefix: list[str]) -> None:
        if query.is_leaf:
            blob = " > ".join(prefix)
            if pattern.search(blob):
                matches.append(prefix)
            return
        for k in query.keys:
            try:
                is_leaf, child = query.descend(k)
            except Exception:
                continue
            if is_leaf:
                blob = " > ".join([*prefix, k])
                if pattern.search(blob) or pattern.search(k):
                    matches.append([*prefix, k])
            else:
                walk(child, [*prefix, k])

    walk(oghma.query(), [])
    return matches


def read_material_by_path(sim_db: Any, db_name: str, path_keys: list[str]) -> Any:
    oghma = sim_db.database(db_name)
    leaf = oghma.walk_query_path(path_keys)
    return read_leaf_material(oghma, leaf)
