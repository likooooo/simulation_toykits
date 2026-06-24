"""Freehand local search Streamlit page."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from filmstack_simulation.materials import RECOMMENDED_SIM_WL_FROM_UM, RECOMMENDED_SIM_WL_TO_UM
from filmstack_simulation.page_widgets import (
    DEFAULT_POLARIZATION,
    init_formula_default,
    init_preset_select,
    on_preset_change,
    panel_head,
    polarization_select,
    range_inputs,
    resolve_stack_cached,
)
from filmstack_simulation.page_styles import inject_filmstack_opt_styles
from filmstack_simulation.filmstack_optimization.local_search.build_config import build_freehand_config
from filmstack_simulation.filmstack_optimization.local_search.freehand_state import FreehandSession, METRICS
from filmstack_simulation.filmstack_optimization.local_search.opt_config import (
    get_freehand_initial_formula,
    get_freehand_initial_preset_id,
    get_freehand_n_wl,
)
from filmstack_simulation.filmstack_optimization.local_search.optimize_runner import run_freehand_optimize
from filmstack_simulation.presets import (
    CUSTOM_PRESET_ID,
    PRESET_IDS,
    PRESET_LABELS,
    build_formula_for_preset,
    get_wl_mid_um,
)
from filmstack_simulation.simulation import compute_rta_at_angle, resolve_stack
from filmstack_simulation.filmstack_optimization.shared.stack_table import stack_table_rows

_FREEHAND_FRONTEND = str(Path(__file__).resolve().parents[1] / "component" / "frontend")
_freehand_component = components.declare_component("freehand_rta_editor", path=_FREEHAND_FRONTEND)


def consume_component_event(
    event: dict[str, Any] | None,
    last_ts: float,
) -> tuple[dict[str, Any] | None, float]:
    """Drop stale replayed events (same or older ts than last consumed)."""
    if event is None:
        return None, last_ts
    ts = float(event.get("ts") or 0)
    if ts <= last_ts:
        return None, last_ts
    return event, ts


def freehand_editor(*, key: str | None = None, **kwargs: Any) -> dict[str, Any] | None:
    return _freehand_component(key=key, default=None, **kwargs)


_TOKENS_PATH = Path(__file__).resolve().parents[1] / "design_tokens.css"
_SESSION_KEY = "fs_opt_freehand"
_FORMULA_KEY = "fs_opt_formula"
_PRESET_KEY = "fs_opt_preset"
_PRESET_SELECT_KEY = "fs_opt_preset_select"
_PAGE_CONTEXT_KEY = "_fs_opt_page_context"
_LAST_EVENT_TS_KEY = "fs_opt_last_event_ts"
_OPT_SUCCESS_KEY = "fs_opt_opt_success"

_WL_FROM_KEY = "fs_opt_wl_from"
_WL_TO_KEY = "fs_opt_wl_to"
_ANGLE_KEY = "fs_opt_angle"
_POLARIZATION_KEY = "fs_opt_polarization"

DEFAULT_WL_FROM = RECOMMENDED_SIM_WL_FROM_UM
DEFAULT_WL_TO = RECOMMENDED_SIM_WL_TO_UM
DEFAULT_ANGLE = 0.0
ANGLE_CLAMP = (-89.9, 89.9)
_BUILD_ROW_COLS = [5, 1]
_BUILD_FORMULA_HEIGHT = 132
_ANGLE_WITH_POL_COLS = [1.15, 1, 0.85]


def inject_global_styles() -> None:
    inject_filmstack_opt_styles(_TOKENS_PATH)


_STACK_RESOLVED_KEY = "fs_opt_stack_resolved"


def _resolve_stack_cached(formula: str, db: Dict[str, Any]) -> tuple[list[Any], list[float]]:
    return resolve_stack_cached(
        formula,
        db,
        cache_key=_STACK_RESOLVED_KEY,
        resolve=resolve_stack,
    )


def _clamp_angle(v: float) -> float:
    return float(max(ANGLE_CLAMP[0], min(ANGLE_CLAMP[1], v)))


def _session() -> FreehandSession:
    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = FreehandSession()
    return st.session_state[_SESSION_KEY]


def _hydrate_widgets_from_session(session: FreehandSession) -> None:
    """Restore widget keys from session after page navigation (keys may be missing)."""
    if not session.built:
        return
    if _FORMULA_KEY not in st.session_state:
        st.session_state[_FORMULA_KEY] = session.working_formula
    if _WL_FROM_KEY not in st.session_state:
        st.session_state[_WL_FROM_KEY] = session.wl_from
    if _WL_TO_KEY not in st.session_state:
        st.session_state[_WL_TO_KEY] = session.wl_to
    if _ANGLE_KEY not in st.session_state:
        st.session_state[_ANGLE_KEY] = session.angle_deg
    if _POLARIZATION_KEY not in st.session_state:
        st.session_state[_POLARIZATION_KEY] = session.polarization


def ensure_session_defaults(
    materials_db: Optional[Dict[str, Any]] = None,
    *,
    sim_wl_from: float | None = None,
    sim_wl_to: float | None = None,
) -> None:
    initial_preset = get_freehand_initial_preset_id()
    init_preset_select(
        preset_key=_PRESET_KEY,
        preset_select_key=_PRESET_SELECT_KEY,
        preset_ids=PRESET_IDS,
        default_preset_id=initial_preset,
    )
    if _FORMULA_KEY not in st.session_state:
        if initial_preset == CUSTOM_PRESET_ID:
            default_formula = get_freehand_initial_formula()
        elif materials_db:
            wl_mid = get_wl_mid_um(sim_wl_from, sim_wl_to)
            default_formula = build_formula_for_preset(
                initial_preset, materials_db, wl_mid
            )
        else:
            default_formula = ""
        init_formula_default(
            formula_key=_FORMULA_KEY,
            materials_db=materials_db,
            default_formula=default_formula,
        )
    if _POLARIZATION_KEY not in st.session_state:
        st.session_state[_POLARIZATION_KEY] = DEFAULT_POLARIZATION


def _on_preset_change() -> None:
    on_preset_change(
        preset_select_key=_PRESET_SELECT_KEY,
        preset_key=_PRESET_KEY,
        formula_key=_FORMULA_KEY,
        page_context_key=_PAGE_CONTEXT_KEY,
        preset_ids=PRESET_IDS,
    )


def _angle_with_polarization_input(
    label: str,
    *,
    key: str,
    default: float,
    fmt: str,
) -> tuple[float, str]:
    title, c_val, c_pol = st.columns(_ANGLE_WITH_POL_COLS, gap="small")
    with title:
        panel_head(label, css_prefix="fs-opt")
    with c_val:
        angle = st.number_input(
            label,
            value=default,
            format=fmt,
            key=key,
            label_visibility="collapsed",
        )
    with c_pol:
        polarization = polarization_select(key=_POLARIZATION_KEY)
    return angle, polarization


GetMaterialsDb = Callable[[], Dict[str, Any]]


@dataclass(frozen=True)
class PageContext:
    get_materials_db: GetMaterialsDb
    sim_wl_from: float | None = None
    sim_wl_to: float | None = None


def _handle_component_event(session: FreehandSession, event: dict[str, Any]) -> bool:
    """Return True if optimization should run."""
    etype = event.get("type")
    if etype == "activeMetric":
        session.active_metric = str(event.get("activeMetric", "R"))
        return False
    if etype == "viewChange":
        session.view_domain = event.get("viewDomain", session.view_domain)
        return False
    if etype == "clearTarget":
        metric = event.get("metric")
        if metric in METRICS:
            session.target[metric] = None
            session.touched[metric] = False
            session.edit_wl_indices[metric] = set()
        return False
    if etype == "curveDragEnd":
        target = event.get("target") or {}
        touched = event.get("touched") or {}
        edit_indices = event.get("editWlIndices") or {}
        metric = event.get("metric")
        for m in METRICS:
            if target.get(m) is not None:
                session.target[m] = np.asarray(target[m], dtype=float)
            if touched.get(m):
                session.touched[m] = True
        if metric in METRICS and edit_indices.get(metric):
            session.edit_wl_indices[metric] |= {int(i) for i in edit_indices[metric]}
        return bool(event.get("triggerOptimize")) and any(session.touched.values())
    return False


def _run_freehand_optimize(
    session: FreehandSession,
    db: Dict[str, Any],
) -> str | None:
    """Run L-BFGS-B synchronously. Return success message, or None on failure."""
    session.optimizing = True
    try:
        with st.spinner("L-BFGS-B 优化中…"):
            cfg = build_freehand_config(
                working_formula=session.working_formula,
                wl_from=session.wl_from,
                wl_to=session.wl_to,
                n_wl=len(session.wl_um),
                angle_deg=session.angle_deg,
                touched=session.touched,
                target=session.target,
                edit_wl_indices=session.edit_wl_indices,
                wl_um=session.wl_um,
                view_domain=session.view_domain,
                polarization=session.polarization,
            )
            opt_formula, merit_history, current, merit_initial = run_freehand_optimize(cfg, db)
            session.apply_optimization_result(
                formula=opt_formula,
                current=current,
                merit_history=merit_history,
                merit_initial=merit_initial,
            )
            st.session_state["fs_opt_reset_component"] = True
            merit_final = merit_history[-1] if merit_history else merit_initial
            return (
                f"优化完成（第 {session.opt_round} 轮），"
                f"merit {merit_initial:.6f} → {merit_final:.6f}"
            )
    except Exception as exc:
        session.optimizing = False
        session.clear_targets()
        st.error(f"优化失败: {exc}")
        import traceback

        st.code(traceback.format_exc())
        return None


def _process_component_event(
    session: FreehandSession,
    event: dict[str, Any],
    db: Dict[str, Any],
) -> bool:
    """Handle component event; run optimization inline when triggered. Return True if rerun needed."""
    should_optimize = _handle_component_event(session, event)
    if not should_optimize:
        return False
    message = _run_freehand_optimize(session, db)
    if message is not None:
        st.session_state[_OPT_SUCCESS_KEY] = message
        return True
    return False


def render_page(
    *,
    context: PageContext,
    materials_db: Optional[Dict[str, Any]] = None,
) -> None:
    st.set_page_config(page_title="Freehand 局部优化", layout="wide")
    inject_global_styles()
    st.session_state[_PAGE_CONTEXT_KEY] = context
    db = materials_db if materials_db is not None else context.get_materials_db()
    ensure_session_defaults(
        db if db else None,
        sim_wl_from=context.sim_wl_from,
        sim_wl_to=context.sim_wl_to,
    )
    session = _session()
    _hydrate_widgets_from_session(session)

    st.markdown('<div class="fs-opt-page-marker"></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="fs-opt-title">Freehand 局部优化</h1>', unsafe_allow_html=True)

    panel_head("多层膜构建指令", css_prefix="fs-opt")
    st.markdown('<div class="fs-opt-build-row-marker"></div>', unsafe_allow_html=True)
    formula_col, action_col = st.columns(_BUILD_ROW_COLS, gap="small")
    with formula_col:
        formula = st.text_area(
            "多层膜构建指令",
            height=_BUILD_FORMULA_HEIGHT,
            label_visibility="collapsed",
            key=_FORMULA_KEY,
        )
    with action_col:
        st.markdown('<div class="fs-opt-build-right-marker"></div>', unsafe_allow_html=True)
        st.selectbox(
            "预设膜系",
            range(len(PRESET_IDS)),
            format_func=lambda i: PRESET_LABELS[PRESET_IDS[i]],
            key=_PRESET_SELECT_KEY,
            on_change=_on_preset_change,
            label_visibility="collapsed",
        )
        build_clicked = st.button("构建", key="fs_opt_build", width="stretch")

    col_wl, _, col_ang = st.columns([1, 0.5, 1], gap="small")
    with col_wl:
        wl_from, wl_to = range_inputs(
            "波长范围 (μm)",
            css_prefix="fs-opt",
            key_from=_WL_FROM_KEY,
            key_to=_WL_TO_KEY,
            default_from=context.sim_wl_from or DEFAULT_WL_FROM,
            default_to=context.sim_wl_to or DEFAULT_WL_TO,
            fmt="%.4f",
        )
    with col_ang:
        st.markdown('<div class="fs-opt-param-right-marker"></div>', unsafe_allow_html=True)
        angle_fix, polarization = _angle_with_polarization_input(
            "固定入射角 θ (°)",
            key=_ANGLE_KEY,
            default=DEFAULT_ANGLE,
            fmt="%.2f",
        )
    angle_fix = _clamp_angle(angle_fix)
    polarization = str(polarization).upper()

    just_built = False
    if build_clicked and formula.strip():
        try:
            materials, thicknesses_um = _resolve_stack_cached(formula.strip(), db)
            curves = compute_rta_at_angle(
                materials,
                thicknesses_um,
                angle_fix,
                wl_from,
                wl_to,
                n_wl=get_freehand_n_wl(),
                polarization=polarization,
            )
            session.reset_after_build(
                formula=formula.strip(),
                wl_um=curves["wl"],
                angle_deg=angle_fix,
                current={"R": curves["R"], "T": curves["T"], "A": curves["A"]},
                wl_from=wl_from,
                wl_to=wl_to,
                polarization=polarization,
            )
            st.session_state["fs_opt_reset_component"] = True
            just_built = True
        except Exception as exc:
            st.error(f"构建失败: {exc}")

    if session.built:
        try:
            materials, thicknesses_um = _resolve_stack_cached(session.working_formula, db)
            st.dataframe(stack_table_rows(materials, thicknesses_um), width="stretch")
        except Exception as exc:
            st.warning(f"层表解析失败: {exc}")

        recompute = False
        if (
            abs(angle_fix - session.angle_deg) > 1e-6
            or polarization != session.polarization
        ):
            recompute = True
        elif (
            abs(wl_from - session.wl_from) > 1e-9
            or abs(wl_to - session.wl_to) > 1e-9
        ):
            recompute = True

        if recompute:
            materials, thicknesses_um = _resolve_stack_cached(session.working_formula, db)
            curves = compute_rta_at_angle(
                materials,
                thicknesses_um,
                angle_fix,
                wl_from,
                wl_to,
                n_wl=len(session.wl_um) if abs(wl_from - session.wl_from) < 1e-9 and abs(wl_to - session.wl_to) < 1e-9 else get_freehand_n_wl(),
                polarization=polarization,
            )
            session.angle_deg = angle_fix
            session.polarization = polarization
            session.wl_from = wl_from
            session.wl_to = wl_to
            session.current = {"R": curves["R"], "T": curves["T"], "A": curves["A"]}
            session.wl_um = curves["wl"]
            session.clear_targets()

        if session.optimizing and not any(session.touched.values()):
            session.optimizing = False

        st.markdown('<div class="fs-opt-section-label">R / T / A vs λ（Freehand）</div>', unsafe_allow_html=True)

        component_args = session.to_component_args()
        component_args["resetTargets"] = bool(
            st.session_state.pop("fs_opt_reset_component", False)
        )

        raw_event = freehand_editor(key="fs_opt_freehand_editor", **component_args)
        needs_rerun = False
        if raw_event is not None and not just_built:
            last_ts = float(st.session_state.get(_LAST_EVENT_TS_KEY, 0.0))
            event, last_ts = consume_component_event(raw_event, last_ts)
            st.session_state[_LAST_EVENT_TS_KEY] = last_ts
            if event is not None:
                needs_rerun = _process_component_event(session, event, db)

        if needs_rerun:
            st.rerun()

        opt_success = st.session_state.pop(_OPT_SUCCESS_KEY, None)
        if opt_success:
            st.success(opt_success)

        st.markdown("**优化后膜系指令**")
        if session.last_optimized_formula:
            st.code(session.last_optimized_formula, language=None)
        else:
            st.caption("完成一次 Freehand 优化后将在此显示膜系指令。")
    elif build_clicked:
        pass
    else:
        st.info("输入膜系公式并点击「构建」以开始 Freehand 优化。")
