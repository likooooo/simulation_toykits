"""Simulation workspace session model (materials + spectrum context)."""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

import streamlit as st

from simulation_database.database_precompiling import get_precompiled_leaf_object
from simulation_database.database_ui import (
    intersect_wl_ranges,
    material_wl_range_um,
    object_unique_name,
    path_id,
    spectrum_wl_range_um,
)

LeafKind = Literal["material", "spectrum"]

_logger = logging.getLogger(__name__)

WORKSPACE_SCHEMA = 5

BROWSER_HELP_TEXT = (
    "双击选中材料加入工作区。\n"
    "单击选中材料进行预览。\n"
    "rii 数据由 https://refractiveindex.info/ 基于 CC0 1.0 协议分享。\n"
    "数据均来自互联网搜集，未检验数据正确性，仅供参考。"
)

WORKSPACE_HELP_TEXT = (
    "工作区数据用于后续模块仿真计算。"
)

RANGE_WARN_HELP_TEXT = (
    "手动调整仿真波长范围后，若某材料或光谱的数据波长区间无法完全覆盖该范围，其卡片会以灰色背景显示。\n"
    "仿真波长范围超出材料波长的部分，nk按材料端点值平坦外推。"
)


@dataclass
class PresentedLeaf:
    kind: LeafKind
    obj: Any
    breadcrumb: str
    path_keys: list[str]
    unique_name: str


@dataclass
class MaterialEntry:
    unique_name: str
    obj: Any
    path_keys: list[str]


@dataclass
class SpectrumEntry:
    unique_name: str
    obj: Any
    breadcrumb: str
    path_keys: list[str]


@dataclass
class FocusEntry:
    kind: LeafKind
    unique_name: str


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


def ensure_sim_workspace() -> SimWorkspace:
    if (
        st.session_state.get("sim_workspace_schema") != WORKSPACE_SCHEMA
        or "sim_workspace" not in st.session_state
    ):
        st.session_state["sim_workspace"] = SimWorkspace()
        st.session_state["sim_workspace_schema"] = WORKSPACE_SCHEMA
        ui = st.session_state.get("sim_workspace_ui")
        if ui is not None:
            ui.defaults_loaded = False
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
    unique_name: str | None = None,
    breadcrumb: str | None = None,
) -> str:
    unique = unique_name or object_unique_name(obj)
    ws.spectra[unique] = SpectrumEntry(
        unique_name=unique,
        obj=obj,
        breadcrumb=breadcrumb if breadcrumb is not None else unique,
        path_keys=list(path_keys),
    )
    ws.last_added_spectrum = unique
    return unique


def add_material_entry(
    ws: SimWorkspace,
    obj: Any,
    path_keys: list[str],
    *,
    unique_name: str | None = None,
) -> str:
    unique = unique_name or object_unique_name(obj)
    ws.materials[unique] = MaterialEntry(
        unique_name=unique,
        obj=obj,
        path_keys=list(path_keys),
    )
    ws.last_added_material = unique
    return unique


def _iter_workspace_entries(ws: SimWorkspace) -> Iterable[tuple[LeafKind, str, MaterialEntry | SpectrumEntry]]:
    for unique_name, entry in ws.spectra.items():
        yield "spectrum", unique_name, entry
    for unique_name, entry in ws.materials.items():
        yield "material", unique_name, entry


def _entry_wl_range_um(kind: LeafKind, entry: MaterialEntry | SpectrumEntry) -> tuple[float, float] | None:
    if kind == "spectrum":
        return spectrum_wl_range_um(entry.obj)
    return material_wl_range_um(entry.obj)


def load_default_workspace_entries(
    sim_db: Any,
    ws: SimWorkspace,
    *,
    material_path_keys: list[list[str]] | None = None,
    spectrum_path_keys: list[list[str]] | None = None,
    strict: bool = False,
    required_material_names: frozenset[str] | None = None,
) -> list[str]:
    """Load default workspace entries; return failed path_id strings."""
    del sim_db  # kept for API compatibility with callers passing sim_db
    failed: list[str] = []
    for path in spectrum_path_keys or []:
        pid = path_id(path)
        try:
            obj = get_precompiled_leaf_object(path)
            add_spectrum_entry(ws, obj, path)
        except Exception as exc:
            failed.append(pid)
            _logger.warning("failed to load default spectrum %s: %s", pid, exc)
    for path in material_path_keys or []:
        pid = path_id(path)
        try:
            obj = get_precompiled_leaf_object(path)
            add_material_entry(ws, obj, path)
        except Exception as exc:
            failed.append(pid)
            _logger.warning("failed to load default material %s: %s", pid, exc)
    if ws.spectra:
        ws.focus = FocusEntry(kind="spectrum", unique_name=next(reversed(ws.spectra)))

    if required_material_names:
        missing = required_material_names - set(ws.materials.keys())
        if missing:
            msg = f"missing required workspace materials: {sorted(missing)}"
            if strict:
                raise RuntimeError(msg)
            _logger.warning(msg)
            failed.extend(f"material:{name}" for name in sorted(missing))

    if strict and failed:
        raise RuntimeError(f"failed to load default workspace entries: {failed}")
    return failed


def ensure_workspace_initialized(
    sim_db: Any,
    *,
    material_path_keys: list[list[str]] | None = None,
    spectrum_path_keys: list[list[str]] | None = None,
    strict: bool = False,
    required_material_names: frozenset[str] | None = None,
) -> tuple[SimWorkspace, SimWorkspaceUI]:
    ws = ensure_sim_workspace()
    ui = ensure_sim_workspace_ui()
    has_paths = bool(material_path_keys or spectrum_path_keys)
    if not ui.defaults_loaded and has_paths:
        load_default_workspace_entries(
            sim_db,
            ws,
            material_path_keys=material_path_keys,
            spectrum_path_keys=spectrum_path_keys,
            strict=strict,
            required_material_names=required_material_names,
        )
        ui.defaults_loaded = True
        refresh_sim_wl_range(ws, ui, force=True)
    return ws, ui


def get_workspace_materials() -> dict[str, Any]:
    return {unique_name: entry.obj for unique_name, entry in ensure_sim_workspace().materials.items()}


def reset_workspace() -> None:
    ws = ensure_sim_workspace()
    ws.materials = OrderedDict()
    ws.spectra = OrderedDict()
    ws.preview = None
    ws.focus = None
    ws.last_added_material = None
    ws.last_added_spectrum = None


def collect_workspace_wl_ranges(ws: SimWorkspace) -> list[tuple[float, float] | None]:
    return [_entry_wl_range_um(kind, entry) for kind, _name, entry in _iter_workspace_entries(ws)]


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
    for kind, unique_name, entry in _iter_workspace_entries(ws):
        data_range = _entry_wl_range_um(kind, entry)
        warnings[f"{kind}:{unique_name}"] = not wl_range_covers_data(
            data_range, ui.sim_wl_from, ui.sim_wl_to
        )
    return warnings


def workspace_to_panel_dict(ws: SimWorkspace, warnings: dict[str, bool] | None = None) -> dict[str, Any]:
    warnings = warnings or {}
    spectra = []
    materials = []
    for kind, unique_name, entry in _iter_workspace_entries(ws):
        item = {
            "unique_name": unique_name,
            "node_path": path_id(entry.path_keys),
            "warn": warnings.get(f"{kind}:{unique_name}", False),
        }
        if kind == "spectrum":
            spectra.append(item)
        else:
            materials.append(item)
    focus = None
    if ws.focus is not None:
        focus = {"kind": ws.focus.kind, "unique_name": ws.focus.unique_name}
    preview = None
    if ws.preview is not None:
        preview = {
            "kind": ws.preview.kind,
            "unique_name": ws.preview.unique_name,
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
