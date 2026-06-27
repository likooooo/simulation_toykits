"""Simulation database page — browse materials/spectra and build workspace."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import streamlit as st
import streamlit.components.v1 as components

import simulation_database_parser as sdp

from simulation_database.database_ui import (
    build_tree_nodes_for_panel,
    dump_object_as_csv,
    infer_leaf_kind,
    material_nk_arrays,
    material_variant_label,
    object_catalog_name,
    object_unique_name,
    path_id,
    search_db_paths,
    spectrum_arrays,
)
from simulation_database.plots import (
    PLOTLY_CHART_CONFIG,
    build_nk_curve_figure,
    build_spectrum_curve_figure,
)
from simulation_database.workspace import (
    BROWSER_HELP_TEXT,
    FocusEntry,
    PresentedLeaf,
    SimWorkspace,
    SimWorkspaceUI,
    add_material_entry,
    add_spectrum_entry,
    ensure_sim_workspace,
    ensure_sim_workspace_ui,
    ensure_workspace_initialized,
    refresh_sim_wl_range,
    reset_workspace,
    workspace_range_warnings,
    workspace_to_panel_dict,
)


_COMPONENT_FRONTEND_DIR = str(Path(__file__).resolve().parent / "component" / "frontend")

_component_func = components.declare_component("simulation_db_panel", path=_COMPONENT_FRONTEND_DIR)


def inject_global_styles(tokens_css: str) -> None:
    """Inject page CSS every rerun — Streamlit discards prior markdown on interaction."""
    st.markdown(
        f"""
        <style>
        {tokens_css}
        html {{ overflow-y: scroll !important; }}
        [data-testid="stAppViewBlockContainer"] {{
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            max-width: 1400px !important;
        }}
        [data-testid="column"] {{
            min-width: 0 !important;
        }}
        .stApp {{
            background: var(--color-bg) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def simulation_db_panel(
    tree_nodes: list[dict[str, Any]],
    expanded_paths: list[str],
    search_mode: bool,
    search_results: list[dict[str, Any]],
    search_query: str,
    workspace: dict[str, Any],
    preview: dict[str, Any] | None,
    status: str = "",
    download_on_action: bool = False,
    browser_help_text: str = "",
    auto_download_base64: str | None = None,
    auto_download_filename: str | None = None,
    height: int = 520,
    section: Literal["browser", "workspace", "all"] = "all",
    tokens_css: str = "",
    key: str | None = None,
) -> dict[str, Any] | None:
    """Render database panel; returns action dict or None."""
    return _component_func(
        tree_nodes=tree_nodes,
        expanded_paths=expanded_paths,
        search_mode=search_mode,
        search_results=search_results,
        search_query=search_query,
        workspace=workspace,
        preview=preview if preview is not None else {},
        status=status,
        download_on_action=download_on_action,
        browser_help_text=browser_help_text or "",
        auto_download_base64=auto_download_base64 or "",
        auto_download_filename=auto_download_filename or "",
        height=height,
        section=section,
        tokens_css=tokens_css,
        key=key,
        default=None,
    )


def encode_download_payload(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


@dataclass
class PanelActionResult:
    status: str = ""
    download_bytes: bytes | None = None
    download_filename: str | None = None
    toast: str = ""


def _load_presented(sim_db: Any, path_keys: list[str]) -> PresentedLeaf:
    obj = sdp.read_at_query_path(sim_db, path_keys)
    kind = infer_leaf_kind(obj)
    catalog = object_catalog_name(obj)
    unique = object_unique_name(obj)
    return PresentedLeaf(
        kind=kind,
        obj=obj,
        breadcrumb=material_variant_label(obj) if kind == "material" else catalog,
        path_keys=list(path_keys),
        catalog_name=catalog,
        unique_name=unique,
    )


def _maybe_dump(obj: Any, download: bool) -> tuple[bytes | None, str | None]:
    if not download:
        return None, None
    return dump_object_as_csv(obj)


def _workspace_object(ws: SimWorkspace, kind: str, catalog_name: str) -> Any | None:
    if kind == "spectrum":
        entry = ws.spectra.get(catalog_name)
        return entry.obj if entry is not None else None
    entry = ws.materials.get(catalog_name)
    return entry.obj if entry is not None else None


WL_FROM_KEY = "sim_db_wl_from"
WL_TO_KEY = "sim_db_wl_to"
WL_DEFAULT_FROM = 0.3
WL_DEFAULT_TO = 2.5


def handle_panel_action(
    sim_db: Any,
    ws: SimWorkspace,
    ui: SimWorkspaceUI,
    action: dict[str, Any],
) -> PanelActionResult:
    result = PanelActionResult()
    act = action.get("action", "")
    workspace_changed = False

    if act == "download_toggle":
        ui.download_on_action = bool(action.get("enabled", False))
        result.status = "下载 CSV：已开启（加入/双击卡片时）" if ui.download_on_action else "下载 CSV：已关闭"
        return result

    if act == "clear_workspace":
        reset_workspace()
        ui.sim_wl_user_set = False
        ws = ensure_sim_workspace()
        refresh_sim_wl_range(ws, ui, force=True)
        if ui.sim_wl_from is not None and ui.sim_wl_to is not None:
            st.session_state[WL_FROM_KEY] = float(ui.sim_wl_from)
            st.session_state[WL_TO_KEY] = float(ui.sim_wl_to)
        else:
            st.session_state[WL_FROM_KEY] = WL_DEFAULT_FROM
            st.session_state[WL_TO_KEY] = WL_DEFAULT_TO
        workspace_changed = True
        result.toast = "已清空工作区"
    elif act == "search":
        ui.search_query = str(action.get("query", "")).strip()
        return result
    elif act == "clear_search":
        ui.search_query = ""
        return result
    elif act == "expand":
        node_path_id = action.get("path_id", "")
        if node_path_id:
            ui.expanded_paths.add(node_path_id)
        return result
    elif act == "collapse":
        node_path_id = action.get("path_id", "")
        if node_path_id:
            ui.expanded_paths.discard(node_path_id)
        return result
    elif act == "remove":
        name = action.get("name", "")
        kind = action.get("kind", "material")
        if kind == "spectrum":
            if name in ws.spectra:
                del ws.spectra[name]
                if ws.last_added_spectrum == name:
                    ws.last_added_spectrum = None
                if ws.focus and ws.focus.kind == "spectrum" and ws.focus.name == name:
                    ws.focus = None
                workspace_changed = True
                result.toast = f"已移除光谱 {name}"
        elif name in ws.materials:
            del ws.materials[name]
            if ws.focus and ws.focus.name == name:
                ws.focus = None
            if ws.last_added_material == name:
                ws.last_added_material = None
            workspace_changed = True
            result.toast = f"已移除 {name}"
    elif act == "focus":
        kind = action.get("kind", "material")
        name = action.get("name", "")
        if kind and name:
            ws.focus = FocusEntry(kind=kind, name=name)
            ws.preview = None
            if action.get("download") and ui.download_on_action:
                obj = _workspace_object(ws, kind, name)
                if obj is not None:
                    zip_b, zip_f = _maybe_dump(obj, True)
                    result.download_bytes = zip_b
                    result.download_filename = zip_f
        return result
    elif act in ("preview", "add"):
        path_keys = action.get("path_keys") or []
        if not path_keys:
            return result
        try:
            presented = _load_presented(sim_db, path_keys)
        except Exception as exc:
            result.status = f"读取失败: {exc}"
            result.toast = result.status
            return result

        if act == "preview":
            ws.preview = presented
            result.status = f"预览: {presented.catalog_name}"
            return result

        if ui.download_on_action:
            zip_b, zip_f = _maybe_dump(presented.obj, True)
            result.download_bytes = zip_b
            result.download_filename = zip_f

        catalog = presented.catalog_name
        if presented.kind == "spectrum":
            add_spectrum_entry(
                ws,
                presented.obj,
                presented.path_keys,
                catalog_name=catalog,
                unique_name=presented.unique_name,
                breadcrumb=presented.catalog_name,
            )
            ws.focus = FocusEntry(kind="spectrum", name=catalog)
        else:
            add_material_entry(
                ws,
                presented.obj,
                presented.path_keys,
                catalog_name=catalog,
                unique_name=presented.unique_name,
            )
            ws.focus = FocusEntry(kind="material", name=catalog)
        ws.preview = None
        workspace_changed = True

    if workspace_changed:
        refresh_sim_wl_range(ws, ui)
    return result


ROOT_BOOTSTRAP_DONE_KEY = "sim_db_roots_bootstrapped"

PANEL_HEIGHT = 720

VIZ_PANEL_ACTIONS = frozenset(
    {"preview", "add", "focus", "remove", "clear_workspace"}
)

VIZ_ACTION_PRIORITY = {
    "add": 5,
    "focus": 4,
    "remove": 3,
    "clear_workspace": 3,
    "preview": 1,
}

TREE_ONLY_ACTIONS = frozenset(
    {"expand", "collapse", "search", "clear_search", "download_toggle"}
)

TREE_CACHE_BUST_ACTIONS = frozenset(
    {
        "expand",
        "collapse",
        "search",
        "clear_search",
        "add",
        "remove",
        "clear_workspace",
    }
)


def _viz_action_rank(action: dict) -> tuple[int, int]:
    act = action.get("action", "")
    return (action.get("ts", 0), VIZ_ACTION_PRIORITY.get(act, 0))


def _bootstrap_root_tree_expansion(sim_db, ws, ui) -> None:
    if st.session_state.get(ROOT_BOOTSTRAP_DONE_KEY):
        return
    if ui.search_query:
        st.session_state[ROOT_BOOTSTRAP_DONE_KEY] = True
        return

    pending: list[tuple[str, list[str]]] = []
    root_query = sim_db.query()
    for key in root_query.keys:
        path_keys = [key]
        root_id = path_id(path_keys)
        if root_id not in ui.expanded_paths:
            pending.append((root_id, path_keys))

    st.session_state[ROOT_BOOTSTRAP_DONE_KEY] = True
    if not pending:
        return

    ts = max(ui.panel_processed_ts, 0) + 1
    for root_id, path_keys in pending:
        _apply_panel_action(
            sim_db,
            ws,
            ui,
            {
                "action": "expand",
                "path_id": root_id,
                "path_keys": path_keys,
                "ts": ts,
            },
        )
        ts += 1
    st.rerun(scope="app")


def _ensure_wl_widget_defaults(ui) -> None:
    if WL_FROM_KEY not in st.session_state:
        st.session_state[WL_FROM_KEY] = float(
            ui.sim_wl_from if ui.sim_wl_from is not None else WL_DEFAULT_FROM
        )
    if WL_TO_KEY not in st.session_state:
        st.session_state[WL_TO_KEY] = float(
            ui.sim_wl_to if ui.sim_wl_to is not None else WL_DEFAULT_TO
        )


def _sync_sim_wl_state(ws, ui) -> tuple[float | None, float | None]:
    _ensure_wl_widget_defaults(ui)
    if ui.sim_wl_user_set:
        ui.sim_wl_from = float(st.session_state[WL_FROM_KEY])
        ui.sim_wl_to = float(st.session_state[WL_TO_KEY])
        return ui.sim_wl_from, ui.sim_wl_to

    refresh_sim_wl_range(ws, ui)
    if ui.sim_wl_from is not None and ui.sim_wl_to is not None:
        st.session_state[WL_FROM_KEY] = float(ui.sim_wl_from)
        st.session_state[WL_TO_KEY] = float(ui.sim_wl_to)
    elif not ws.spectra and not ws.materials:
        ui.sim_wl_from = WL_DEFAULT_FROM
        ui.sim_wl_to = WL_DEFAULT_TO
        st.session_state[WL_FROM_KEY] = WL_DEFAULT_FROM
        st.session_state[WL_TO_KEY] = WL_DEFAULT_TO
    return ui.sim_wl_from, ui.sim_wl_to


def _mark_wl_user_set() -> None:
    ensure_sim_workspace_ui().sim_wl_user_set = True


def _plot_wl_bounds(sim_from: float | None, sim_to: float | None) -> tuple[float, float]:
    from_val = sim_from if sim_from is not None else float(st.session_state.get(WL_FROM_KEY, WL_DEFAULT_FROM))
    to_val = sim_to if sim_to is not None else float(st.session_state.get(WL_TO_KEY, WL_DEFAULT_TO))
    return from_val, to_val


def _tree_cache_key(ui) -> tuple:
    return (frozenset(ui.expanded_paths), ui.search_query)


def _tree_nodes_for_panel(sim_db, ui) -> list:
    key = _tree_cache_key(ui)
    if ui.tree_cache_key == key and ui.tree_nodes_cache is not None:
        return ui.tree_nodes_cache
    nodes = build_tree_nodes_for_panel(sim_db, ui.expanded_paths, ui.children_cache)
    ui.tree_cache_key = key
    ui.tree_nodes_cache = nodes
    return nodes


def _build_panel_common(sim_db, ws, ui) -> dict:
    tree_nodes = _tree_nodes_for_panel(sim_db, ui)
    search_query = ui.search_query
    search_mode = bool(search_query)
    search_results = search_db_paths(sim_db, search_query) if search_mode else []
    warnings = workspace_range_warnings(ws, ui)
    workspace_dict = workspace_to_panel_dict(ws, warnings)
    preview_dict = workspace_dict.get("preview") or {}
    return dict(
        tree_nodes=tree_nodes,
        expanded_paths=list(ui.expanded_paths),
        search_mode=search_mode,
        search_results=search_results,
        search_query=search_query,
        workspace=workspace_dict,
        preview=preview_dict,
        status=ui.panel_status,
        download_on_action=ui.download_on_action,
        browser_help_text=BROWSER_HELP_TEXT,
        height=PANEL_HEIGHT,
    )


def _take_download_payload(ui) -> tuple[str, str]:
    token = ui.auto_download_token
    if not token:
        return "", ""
    ui.auto_download_token = None
    return token.get("base64", ""), token.get("filename", "")


def _apply_panel_action(sim_db, ws, ui, panel_action: dict) -> None:
    act = panel_action.get("action", "")
    ui.panel_processed_ts = panel_action.get("ts", 0)
    if act in VIZ_PANEL_ACTIONS:
        ui.viz_rev += 1

    result = handle_panel_action(sim_db, ws, ui, panel_action)
    if act in TREE_CACHE_BUST_ACTIONS:
        ui.tree_cache_key = None
        ui.tree_nodes_cache = None
    if result.toast:
        st.toast(result.toast)
    if result.status:
        ui.panel_status = result.status
    if result.download_bytes and result.download_filename:
        ui.auto_download_token = {
            "base64": encode_download_payload(result.download_bytes),
            "filename": result.download_filename,
        }
        st.toast(f"已开始下载：{result.download_filename}")


def _is_new_panel_action(ui, panel_action: dict | None) -> bool:
    if not panel_action or not isinstance(panel_action, dict):
        return False
    action_ts = panel_action.get("ts", 0)
    return bool(action_ts and action_ts > ui.panel_processed_ts)


def _process_panel_actions(
    sim_db,
    ws,
    ui,
    browser_action: dict | None,
    workspace_action: dict | None,
) -> None:
    candidates: list[dict] = []
    for action in (browser_action, workspace_action):
        if not _is_new_panel_action(ui, action):
            continue
        candidates.append(action)
    if not candidates:
        return
    panel_action = max(candidates, key=_viz_action_rank)
    act = panel_action.get("action", "")
    if act in TREE_ONLY_ACTIONS or act in VIZ_PANEL_ACTIONS:
        _apply_panel_action(sim_db, ws, ui, panel_action)
        st.rerun(scope="app")


def render_page(*, tokens_path: Path) -> None:
    """Render the simulation database three-column page."""
    tokens_css = tokens_path.read_text(encoding="utf-8")
    st.set_page_config(page_title="仿真数据库", layout="wide")
    inject_global_styles(tokens_css)

    st.markdown(
        """
<style>
[data-testid="stAppViewContainer"]:has(.sim-db-page-marker) [data-testid="block-container"] {
  padding-top: 0.5rem !important;
  padding-bottom: 0.5rem !important;
  padding-left: 0.75rem !important;
  padding-right: 0.75rem !important;
  max-width: none !important;
}
.sim-db-title {
  font-family: var(--font-ui);
  font-size: 1.125rem;
  font-weight: 600;
  color: var(--color-text);
  margin: 0 0 0.5rem 0;
  line-height: 1.2;
}
[data-testid="stAppViewContainer"]:has(.sim-db-page-marker) [data-testid="stHorizontalBlock"] {
  align-items: flex-start;
  gap: 0.5rem;
}
[data-testid="stAppViewContainer"]:has(.sim-db-page-marker) [data-testid="column"] {
  min-height: calc(100dvh - 4.5rem);
}
[data-testid="stAppViewContainer"]:has(.sim-db-page-marker) [data-testid="stCustomComponentV1"] {
  margin: 0 !important;
}
[data-testid="stAppViewContainer"]:has(.sim-db-page-marker) [data-testid="stCustomComponentV1"] iframe {
  min-height: calc(100dvh - 4.5rem) !important;
  height: calc(100dvh - 4.5rem) !important;
  display: block;
}
[data-testid="stAppViewContainer"]:has(.sim-db-page-marker) [data-testid="column"]:has(.sim-db-viz-marker) {
  min-width: 0;
  background: var(--color-surface, #fff);
  border: 1px solid var(--color-border, #E5E7EB);
  border-radius: 10px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06);
  overflow: hidden;
}
.sim-db-viz-marker {
  display: none;
}
[data-testid="stAppViewContainer"]:has(.sim-db-page-marker) [data-testid="column"]:has(.sim-db-viz-marker) > div > [data-testid="stVerticalBlock"] {
  padding: 12px;
  gap: 0.35rem;
}
[data-testid="stAppViewContainer"]:has(.sim-db-page-marker) [data-testid="column"]:has(.sim-db-viz-marker) [data-testid="element-container"] {
  margin-bottom: 0.1rem;
}
.sim-db-panel-head {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--color-text-secondary, #6B7280);
  margin: 0;
  line-height: 2.25rem;
  white-space: nowrap;
}
[data-testid="stAppViewContainer"]:has(.sim-db-page-marker) [data-testid="column"]:has(.sim-db-viz-marker) [data-testid="stNumberInput"] {
  margin-bottom: 0;
}
[data-testid="stAppViewContainer"]:has(.sim-db-page-marker) [data-testid="column"]:has(.sim-db-viz-marker) [data-testid="stNumberInput"] input {
  padding-top: 0.2rem;
  padding-bottom: 0.2rem;
  min-height: 2rem;
}
.sim-db-wl-arrow {
  text-align: center;
  padding-top: 0.45rem;
  color: #6B7280;
  line-height: 1;
}
[data-testid="stAppViewContainer"]:has(.sim-db-page-marker) [data-testid="column"]:has(.sim-db-viz-marker) [data-testid="stPlotlyChart"] {
  min-height: 220px;
}
</style>
<div class="sim-db-page-marker"></div>
<h1 class="sim-db-title">仿真数据库</h1>
""",
        unsafe_allow_html=True,
    )

    sim_db = sdp.get_simulation_database(init=True)
    ws, ui = ensure_workspace_initialized(sim_db)
    _bootstrap_root_tree_expansion(sim_db, ws, ui)
    sim_from, sim_to = _sync_sim_wl_state(ws, ui)
    plot_from, plot_to = _plot_wl_bounds(sim_from, sim_to)

    auto_download_base64, auto_download_filename = _take_download_payload(ui)

    col_browser, col_workspace, col_viz = st.columns(3, gap="small")

    _panel_kwargs = _build_panel_common(sim_db, ws, ui)
    _panel_kwargs["tokens_css"] = tokens_css

    with col_browser:
        panel_action_browser = simulation_db_panel(
            **_panel_kwargs,
            auto_download_base64=auto_download_base64,
            auto_download_filename=auto_download_filename,
            section="browser",
            key="simulation_db_panel_browser",
        )

    with col_workspace:
        panel_action_workspace = simulation_db_panel(
            **_panel_kwargs,
            auto_download_base64=auto_download_base64,
            auto_download_filename=auto_download_filename,
            section="workspace",
            key="simulation_db_panel_workspace",
        )

    _process_panel_actions(
        sim_db,
        ws,
        ui,
        panel_action_browser,
        panel_action_workspace,
    )

    with col_viz:
        st.markdown('<div class="sim-db-viz-marker"></div>', unsafe_allow_html=True)

        wl_title, wl_c1, wl_mid, wl_c2, _ = st.columns([1.15, 1, 0.06, 1, 0.02], gap="small")
        with wl_title:
            st.markdown('<div class="sim-db-panel-head">仿真波长范围 (μm)</div>', unsafe_allow_html=True)
        with wl_c1:
            st.number_input(
                "from",
                format="%.4f",
                key=WL_FROM_KEY,
                label_visibility="collapsed",
                on_change=_mark_wl_user_set,
            )
        with wl_mid:
            st.markdown('<div class="sim-db-wl-arrow">→</div>', unsafe_allow_html=True)
        with wl_c2:
            st.number_input(
                "to",
                format="%.4f",
                key=WL_TO_KEY,
                label_visibility="collapsed",
                on_change=_mark_wl_user_set,
            )

        spec_plot_slot = st.empty()
        mat_plot_slot = st.empty()

        def _resolve_spectrum_plot():
            preview = ws.preview
            if preview and preview.kind == "spectrum":
                return preview.obj, preview.catalog_name, True
            if ws.focus and ws.focus.kind == "spectrum" and ws.focus.name in ws.spectra:
                entry = ws.spectra[ws.focus.name]
                return entry.obj, ws.focus.name, False
            if ws.last_added_spectrum and ws.last_added_spectrum in ws.spectra:
                entry = ws.spectra[ws.last_added_spectrum]
                return entry.obj, ws.last_added_spectrum, False
            if ws.spectra:
                catalog = next(reversed(ws.spectra))
                entry = ws.spectra[catalog]
                return entry.obj, catalog, False
            return None, None, False

        def _resolve_material_plot():
            preview = ws.preview
            if preview and preview.kind == "material":
                return preview.obj, preview.catalog_name, True
            if ws.focus and ws.focus.kind == "material" and ws.focus.name in ws.materials:
                mat = ws.materials[ws.focus.name].obj
                return mat, ws.focus.name, False
            if ws.last_added_material and ws.last_added_material in ws.materials:
                mat = ws.materials[ws.last_added_material].obj
                return mat, ws.last_added_material, False
            return None, None, False

        focus_key = ws.focus.name if ws.focus else "none"
        spec_focus_key = focus_key if ws.focus and ws.focus.kind == "spectrum" else "none"
        mat_focus_key = focus_key if ws.focus and ws.focus.kind == "material" else "none"
        spec_preview_key = (
            ws.preview.catalog_name if ws.preview and ws.preview.kind == "spectrum" else "none"
        )
        mat_preview_key = (
            ws.preview.catalog_name if ws.preview and ws.preview.kind == "material" else "none"
        )
        plot_rev = ui.viz_rev

        spec_obj, spec_key, spec_is_preview = _resolve_spectrum_plot()
        mat_obj, mat_key, mat_is_preview = _resolve_material_plot()

        chart_h = max(240, (PANEL_HEIGHT - 110) // 2)

        if spec_obj is not None:
            try:
                wl, val = spectrum_arrays(spec_obj)
                spec_title = spec_key if spec_key else "光谱"
                if spec_is_preview:
                    spec_title = f"{spec_title}（预览）"
                fig = build_spectrum_curve_figure(
                    wl, val, title=spec_title, sim_wl_from=plot_from, sim_wl_to=plot_to, height=chart_h
                )
                spec_plot_slot.plotly_chart(
                    fig,
                    width="stretch",
                    key=f"spec_{spec_key}_{spec_focus_key}_{spec_preview_key}_{plot_rev}_{plot_from}_{plot_to}",
                    config=PLOTLY_CHART_CONFIG,
                )
            except Exception as e:
                spec_plot_slot.error(f"光谱数据获取失败: {e}")
        else:
            spec_plot_slot.empty()

        if mat_obj is not None:
            try:
                wl, n_vals, k_vals = material_nk_arrays(mat_obj)
                path_hash = hashlib.md5(str(mat_key).encode()).hexdigest()[:8]
                mat_title = mat_key if mat_key else "材料"
                if mat_is_preview:
                    mat_title = f"{mat_title}（预览）"
                fig = build_nk_curve_figure(
                    wl, n_vals, k_vals, title=mat_title, sim_wl_from=plot_from, sim_wl_to=plot_to, height=chart_h
                )
                mat_plot_slot.plotly_chart(
                    fig,
                    width="stretch",
                    key=f"mat_{path_hash}_{mat_focus_key}_{mat_preview_key}_{plot_rev}_{plot_from}_{plot_to}",
                    config=PLOTLY_CHART_CONFIG,
                )
            except Exception as e:
                mat_plot_slot.error(f"材料数据获取失败: {e}")
        else:
            mat_plot_slot.empty()
