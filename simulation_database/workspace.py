"""Simulation workspace session model (materials + spectrum context)."""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Literal

import streamlit as st

from simulation_database.database_ui import (
    DEFAULT_SPECTRUM_PATH,
    browser_node_path,
    intersect_wl_ranges,
    material_wl_range_um,
    object_catalog_name,
    object_unique_name,
    read_leaf_at_path,
    spectrum_wl_range_um,
)

LeafKind = Literal["material", "spectrum"]

_logger = logging.getLogger(__name__)

WORKSPACE_SCHEMA = 3

BROWSER_HELP_TEXT = (
    "materials/refractive_index_info 数据由 https://refractiveindex.info/ 基于 CC0 1.0 协议分享。\n"
    "其它数据来自互联网，数据仅供参考。"
)

WORKSPACE_HELP_TEXT = (
    "在左侧「资源浏览器」中双击 leaf 节点可加入工作区。\n"
    "工作区数据供 Filmstack 等模块使用。"
)

RANGE_WARN_HELP_TEXT = (
    "手动调整仿真波长范围后，若某材料或光谱的数据波长区间无法完全覆盖该范围，"
    "其卡片会以灰色背景显示。仿真波长范围超出材料波长的部分，按材料端点值平坦外推。"
)


@dataclass
class PresentedLeaf:
    kind: LeafKind
    obj: Any
    breadcrumb: str
    path_keys: list[str]
    db_name: str
    catalog_name: str
    unique_name: str


@dataclass
class MaterialEntry:
    catalog_name: str
    unique_name: str
    obj: Any
    path_keys: list[str]
    db_name: str = "materials"


@dataclass
class SpectrumEntry:
    catalog_name: str
    unique_name: str
    obj: Any
    breadcrumb: str
    path_keys: list[str]
    db_name: str = "spectra"


@dataclass
class FocusEntry:
    kind: LeafKind
    name: str


@dataclass
class SimWorkspace:
    materials: OrderedDict[str, MaterialEntry] = field(default_factory=OrderedDict)
    spectra: OrderedDict[str, SpectrumEntry] = field(default_factory=OrderedDict)
    preview: PresentedLeaf | None = None
    focus: FocusEntry | None = None
    last_added_material: str | None = None
    last_added_spectrum: str | None = None


@dataclass
class SimWorkspaceUI:
    expanded_paths: set[str] = field(default_factory=set)
    children_cache: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    search_query: str = ""
    download_on_action: bool = False
    panel_status: str = ""
    panel_processed_ts: int = 0
    viz_rev: int = 0
    auto_download_token: dict[str, str] | None = None
    sim_wl_from: float | None = None
    sim_wl_to: float | None = None
    sim_wl_user_set: bool = False
    defaults_loaded: bool = False
    tree_cache_key: tuple | None = None
    tree_nodes_cache: list[dict[str, Any]] | None = None


def _default_workspace() -> SimWorkspace:
    return SimWorkspace()


def _coerce_material_entry(catalog_name: str, value: Any) -> MaterialEntry:
    if isinstance(value, MaterialEntry):
        return value
    return MaterialEntry(
        catalog_name=catalog_name,
        unique_name=object_unique_name(value),
        obj=value,
        path_keys=[],
        db_name="materials",
    )


def _migrate_workspace(ws: Any) -> SimWorkspace:
    if isinstance(ws, SimWorkspace) and hasattr(ws, "spectra"):
        migrated = OrderedDict(
            (name, _coerce_material_entry(name, entry))
            for name, entry in ws.materials.items()
        )
        ws.materials = migrated
        return ws
    fresh = _default_workspace()
    if ws is not None:
        if hasattr(ws, "materials"):
            for name, entry in getattr(ws, "materials", {}).items():
                fresh.materials[name] = _coerce_material_entry(name, entry)
        legacy = getattr(ws, "spectrum", None)
        if legacy is not None:
            fresh.spectra[legacy.catalog_name] = legacy
    return fresh


def ensure_sim_workspace() -> SimWorkspace:
    schema = st.session_state.get("sim_workspace_schema")
    if schema != WORKSPACE_SCHEMA or "sim_workspace" not in st.session_state:
        ws = _migrate_workspace(st.session_state.get("sim_workspace"))
        st.session_state["sim_workspace"] = ws
        st.session_state["sim_workspace_schema"] = WORKSPACE_SCHEMA
        ui = st.session_state.get("sim_workspace_ui")
        if ui is not None:
            ui.defaults_loaded = False
        return ws
    return st.session_state["sim_workspace"]


def ensure_sim_workspace_ui() -> SimWorkspaceUI:
    if "sim_workspace_ui" not in st.session_state:
        st.session_state["sim_workspace_ui"] = SimWorkspaceUI()
    return st.session_state["sim_workspace_ui"]


def add_spectrum_entry(
    ws: SimWorkspace,
    obj: Any,
    path_keys: list[str],
    *,
    catalog_name: str | None = None,
    unique_name: str | None = None,
    breadcrumb: str | None = None,
) -> str:
    catalog = catalog_name or object_catalog_name(obj)
    ws.spectra[catalog] = SpectrumEntry(
        catalog_name=catalog,
        unique_name=unique_name or object_unique_name(obj),
        obj=obj,
        breadcrumb=breadcrumb if breadcrumb is not None else catalog,
        path_keys=list(path_keys),
        db_name="spectra",
    )
    ws.last_added_spectrum = catalog
    return catalog


def add_material_entry(
    ws: SimWorkspace,
    obj: Any,
    path_keys: list[str],
    *,
    catalog_name: str | None = None,
    unique_name: str | None = None,
) -> str:
    catalog = catalog_name or object_catalog_name(obj)
    ws.materials[catalog] = MaterialEntry(
        catalog_name=catalog,
        unique_name=unique_name or object_unique_name(obj),
        obj=obj,
        path_keys=list(path_keys),
        db_name="materials",
    )
    ws.last_added_material = catalog
    return catalog


def _try_add_spectrum(sim_db: Any, ws: SimWorkspace, path_keys: list[str]) -> None:
    try:
        spec_obj = read_leaf_at_path(sim_db, "spectra", path_keys)
        add_spectrum_entry(ws, spec_obj, path_keys)
    except Exception as exc:
        _logger.debug("failed to load default spectrum %s: %s", path_keys, exc)


def _try_add_material(sim_db: Any, ws: SimWorkspace, path_keys: list[str]) -> None:
    try:
        mat_obj = read_leaf_at_path(sim_db, "materials", path_keys)
        add_material_entry(ws, mat_obj, path_keys)
    except Exception as exc:
        _logger.debug("failed to load default material %s: %s", path_keys, exc)


def load_default_workspace_entries(
    sim_db: Any,
    ws: SimWorkspace,
    *,
    material_path_keys: list[list[str]] | None = None,
    spectrum_path_keys: list[list[str]] | None = None,
) -> None:
    spectrum_paths = spectrum_path_keys if spectrum_path_keys is not None else [DEFAULT_SPECTRUM_PATH]
    for path in spectrum_paths:
        _try_add_spectrum(sim_db, ws, path)
    for path in material_path_keys or []:
        _try_add_material(sim_db, ws, path)
    if ws.spectra:
        ws.focus = FocusEntry(kind="spectrum", name=next(reversed(ws.spectra)))


def ensure_workspace_initialized(
    sim_db: Any,
    *,
    material_path_keys: list[list[str]] | None = None,
    spectrum_path_keys: list[list[str]] | None = None,
) -> tuple[SimWorkspace, SimWorkspaceUI]:
    ws = ensure_sim_workspace()
    ui = ensure_sim_workspace_ui()
    if not ui.defaults_loaded:
        load_default_workspace_entries(
            sim_db,
            ws,
            material_path_keys=material_path_keys,
            spectrum_path_keys=spectrum_path_keys,
        )
        ui.defaults_loaded = True
        refresh_sim_wl_range(ws, ui, force=True)
    return ws, ui


def get_workspace_materials() -> dict[str, Any]:
    return {name: entry.obj for name, entry in ensure_sim_workspace().materials.items()}


def reset_workspace() -> None:
    ws = ensure_sim_workspace()
    ws.materials = OrderedDict()
    ws.spectra = OrderedDict()
    ws.preview = None
    ws.focus = None
    ws.last_added_material = None
    ws.last_added_spectrum = None


def collect_workspace_wl_ranges(ws: SimWorkspace) -> list[tuple[float, float] | None]:
    ranges: list[tuple[float, float] | None] = []
    for entry in ws.spectra.values():
        ranges.append(spectrum_wl_range_um(entry.obj))
    for entry in ws.materials.values():
        ranges.append(material_wl_range_um(entry.obj))
    return ranges


def refresh_sim_wl_range(ws: SimWorkspace, ui: SimWorkspaceUI, *, force: bool = False) -> None:
    if ui.sim_wl_user_set and not force:
        return
    intersection = intersect_wl_ranges(collect_workspace_wl_ranges(ws))
    if intersection is None:
        ui.sim_wl_from = None
        ui.sim_wl_to = None
        return
    ui.sim_wl_from, ui.sim_wl_to = intersection


def wl_range_covers_data(data_range: tuple[float, float] | None, sim_from: float | None, sim_to: float | None) -> bool:
    if data_range is None or sim_from is None or sim_to is None:
        return True
    sim_lo = min(sim_from, sim_to)
    sim_hi = max(sim_from, sim_to)
    lo, hi = data_range
    return lo <= sim_lo and hi >= sim_hi


def workspace_range_warnings(ws: SimWorkspace, ui: SimWorkspaceUI) -> dict[str, bool]:
    if not ui.sim_wl_user_set:
        return {}
    warnings: dict[str, bool] = {}
    for catalog_name, entry in ws.spectra.items():
        spec_range = spectrum_wl_range_um(entry.obj)
        warnings[f"spectrum:{catalog_name}"] = not wl_range_covers_data(
            spec_range, ui.sim_wl_from, ui.sim_wl_to
        )
    for catalog_name, entry in ws.materials.items():
        mat_range = material_wl_range_um(entry.obj)
        warnings[f"material:{catalog_name}"] = not wl_range_covers_data(
            mat_range, ui.sim_wl_from, ui.sim_wl_to
        )
    return warnings


def workspace_to_panel_dict(ws: SimWorkspace, warnings: dict[str, bool] | None = None) -> dict[str, Any]:
    warnings = warnings or {}
    spectra = []
    for catalog_name, entry in ws.spectra.items():
        spectra.append(
            {
                "catalog_name": catalog_name,
                "unique_name": entry.unique_name,
                "name": catalog_name,
                "node_path": browser_node_path(entry.db_name, entry.path_keys),
                "warn": warnings.get(f"spectrum:{catalog_name}", False),
            }
        )
    materials = []
    for catalog_name, entry in ws.materials.items():
        materials.append(
            {
                "catalog_name": catalog_name,
                "unique_name": entry.unique_name,
                "name": catalog_name,
                "node_path": browser_node_path(entry.db_name, entry.path_keys),
                "warn": warnings.get(f"material:{catalog_name}", False),
            }
        )
    focus = None
    if ws.focus is not None:
        focus = {"kind": ws.focus.kind, "name": ws.focus.name}
    preview = None
    if ws.preview is not None:
        preview = {
            "kind": ws.preview.kind,
            "catalog_name": ws.preview.catalog_name,
            "unique_name": ws.preview.unique_name,
            "name": ws.preview.catalog_name,
            "breadcrumb": ws.preview.breadcrumb,
        }
    return {
        "spectra": spectra,
        "materials": materials,
        "focus": focus,
        "preview": preview,
        "help_text": WORKSPACE_HELP_TEXT,
        "range_warn_text": RANGE_WARN_HELP_TEXT,
    }
