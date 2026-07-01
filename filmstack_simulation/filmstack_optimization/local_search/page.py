"""Freehand local search Streamlit page."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from filmstack_simulation.help_texts import (
    FORMULA_DOCS_URL,
    FORMULA_HELP_TEXT,
    FREEHAND_CHART_HELP_TEXT,
    OPTIMIZED_FORMULA_HELP_TEXT,
    WL_RANGE_LABEL,
)
from filmstack_simulation.page_widgets import (
    OPT_FORMULA_STACK_HEIGHT_PX,
    OPT_UI_APPLY,
    DEFAULT_FIXED_ANGLE,
    angle_polarization_inputs,
    make_preset_change_handler,
    opt_ui_defaults,
    panel_head,
    params_stack_spacer,
    preset_select_row,
    range_inputs,
    resolve_stack_cached,
    show_rebuild_prompt,
)
import filmstack_visualizer
import filmstack_optimization_utils as fos
from filmstack_simulation.page_shell import (
    FilmstackSessionKeys,
    PageContext,
    bootstrap_filmstack_page,
    ensure_filmstack_session_defaults,
)
from filmstack_simulation.page_styles import inject_filmstack_opt_styles
from filmstack_simulation.filmstack_optimization.local_search.freehand_state import (
    FreehandSession,
    METRICS,
    build_freehand_wl_indices,
    clamp_freehand_target_array,
    validate_freehand_targets,
)
from filmstack_simulation.filmstack_optimization.local_search.opt_config import (
    get_freehand_cost_scope,
    get_freehand_default_thickness_range_pct,
    get_freehand_n_wl,
    load_freehand_base_config,
)
from filmstack_simulation.filmstack_optimization.shared.stack_table import (
    apply_optimized_thicknesses_to_formula,
    film_layer_indices,
    layer_bounds_from_ranges,
    stack_table_rows,
    sync_layer_range_pct_from_table,
)
from filmstack_simulation.presets import PresetCatalog
from filmstack_simulation.simulation import compute_rta_at_angle, resolve_stack, resolve_stack_with_layers

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


_SESSION_KEY = "fs_opt_freehand"
_FORMULA_KEY = "fs_opt_formula"
_PRESET_KEY = "fs_opt_preset"
_PRESET_SELECT_KEY = "fs_opt_preset_select"
_PAGE_CONTEXT_KEY = "_fs_opt_page_context"
_LAST_EVENT_TS_KEY = "fs_opt_last_event_ts"
_RENDER_GEN_KEY = "fs_opt_render_gen"
_OPT_SUCCESS_KEY = "fs_opt_opt_success"
_LAYER_TABLE_KEY = "fs_opt_layer_table"
_VIEW_UI_EVENTS = frozenset({"viewChange", "activeMetric", "clearTarget"})

_WL_FROM_KEY = "fs_opt_wl_from"
_WL_TO_KEY = "fs_opt_wl_to"
_ANGLE_KEY = "fs_opt_angle"
_POLARIZATION_KEY = "fs_opt_polarization"
_POLARIZATION_CHANGED_KEY = "fs_opt_polarization_changed"
_PRESET_CHANGED_KEY = "fs_opt_preset_changed"

_ANGLE_CLAMP = (-89.9, 89.9)
_INPUT_ROW_COLS = [1.5, 1]


_SESSION_KEYS = FilmstackSessionKeys(
    formula_key=_FORMULA_KEY,
    preset_key=_PRESET_KEY,
    preset_select_key=_PRESET_SELECT_KEY,
    polarization_key=_POLARIZATION_KEY,
    page_context_key=_PAGE_CONTEXT_KEY,
)


def build_freehand_config(
    *,
    working_formula: str,
    wl_from: float,
    wl_to: float,
    n_wl: int,
    angle_deg: float,
    touched: Mapping[str, bool],
    target: Mapping[str, np.ndarray | None],
    wl_um: np.ndarray | None = None,
    view_domain: Mapping[str, dict[str, list[float]]] | None = None,
    edit_wl_indices: Mapping[str, set[int] | list[int] | None] | None = None,
    cost_scope: str | None = None,
    polarization: str = "UNPOLARIZED",
    film_indices: Sequence[int] | None = None,
    thicknesses_um: Sequence[float] | None = None,
    layer_range_pct: Mapping[int, float] | None = None,
) -> Dict[str, Any]:
    wl_step = (float(wl_to) - float(wl_from)) / max(int(n_wl) - 1, 1)
    scope = cost_scope if cost_scope is not None else get_freehand_cost_scope()
    runtime: Dict[str, Any] = {
        "formula": working_formula,
        "target_wl": [float(wl_from), float(wl_to), wl_step],
        "target_angle": [float(angle_deg), float(angle_deg)],
        "polarization": str(polarization).upper(),
        "freehand_touched": {k: bool(touched.get(k)) for k in ("R", "T", "A")},
        "freehand_cost_scope": scope,
    }
    if touched.get("R") and target.get("R") is not None:
        runtime["R_target_spectrum"] = np.asarray(target["R"], dtype=float).reshape(1, -1).tolist()
    if touched.get("T") and target.get("T") is not None:
        runtime["T_target_spectrum"] = np.asarray(target["T"], dtype=float).reshape(1, -1).tolist()
    if touched.get("A") and target.get("A") is not None:
        runtime["A_target_spectrum"] = np.asarray(target["A"], dtype=float).reshape(1, -1).tolist()
    validate_freehand_targets(touched, target)
    if wl_um is not None:
        wl_indices = build_freehand_wl_indices(
            scope=scope,
            wl_um=np.asarray(wl_um, dtype=float),
            touched=touched,
            view_domain=view_domain or {},
            edit_wl_indices=edit_wl_indices or {},
        )
        if wl_indices:
            runtime["freehand_wl_indices"] = wl_indices
    if (
        film_indices is not None
        and thicknesses_um is not None
        and layer_range_pct is not None
    ):
        runtime["layer_bounds"] = layer_bounds_from_ranges(
            film_indices, thicknesses_um, layer_range_pct
        )
    return filmstack_visualizer.merge_filmstack_optimization_config(
        load_freehand_base_config(), runtime
    )


def run_freehand_optimize(
    cfg: Dict[str, Any],
    materials_db: Dict[str, Any],
) -> tuple[str, list[float], Dict[str, np.ndarray], float]:
    spec = fos.stack_from_formula(cfg["formula"], materials_db)
    targets, target_wls, target_angles = fos.build_targets_from_cfg(cfg)
    pol = fos.Polarization(str(cfg["polarization"]).upper())
    ctx = fos.make_objective_context(spec, targets, pol, cfg)
    ctx.freehand_touched = dict(cfg.get("freehand_touched", {}))
    ctx.freehand_wl_indices = dict(cfg.get("freehand_wl_indices", {}))
    ctx.optimization_cfg = cfg

    cost_fn = fos.load_filmstack_cost_function(
        cfg["cost_function"]["path"], cfg["cost_function"]["name"]
    )
    x0 = np.array([spec.thicknesses_um[i] for i in spec.film_indices], dtype=float)
    merit_initial, _ = cost_fn(x0, ctx)
    opt_x, _, merit_history, _ = fos._run_optimize(spec, ctx, cost_fn, cfg)

    thicknesses = list(spec.thicknesses_um)
    for idx, t in zip(spec.film_indices, opt_x):
        thicknesses[idx] = float(t)
    optimized_formula = apply_optimized_thicknesses_to_formula(
        cfg["formula"], spec.film_indices, thicknesses
    )

    wls, angles, _, _ = fos.resolve_target_axes(cfg)
    curves = compute_rta_at_angle(
        spec.materials,
        thicknesses,
        float(angles[0]),
        float(wls.min()),
        float(wls.max()),
        n_wl=len(wls),
        polarization=str(cfg.get("polarization", "UNPOLARIZED")),
    )
    current = {"R": curves["R"], "T": curves["T"], "A": curves["A"]}
    return optimized_formula, merit_history, current, float(merit_initial)


_STACK_RESOLVED_KEY = "fs_opt_stack_resolved"


def _resolve_stack_cached(formula: str, db: Dict[str, Any]) -> tuple[list[Any], list[float]]:
    return resolve_stack_cached(
        formula,
        db,
        cache_key=_STACK_RESOLVED_KEY,
        resolve=resolve_stack,
    )


def _clamp_angle(v: float) -> float:
    return float(max(_ANGLE_CLAMP[0], min(_ANGLE_CLAMP[1], v)))


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
    *,
    preset_catalog: PresetCatalog,
    initial_preset_id: str,
    initial_formula: str,
) -> None:
    ensure_filmstack_session_defaults(
        keys=_SESSION_KEYS,
        preset_catalog=preset_catalog,
        initial_preset_id=initial_preset_id,
        initial_formula=initial_formula,
    )


def _on_polarization_change() -> None:
    st.session_state[_POLARIZATION_CHANGED_KEY] = True


def _on_preset_change() -> None:
    make_preset_change_handler(
        preset_changed_key=_PRESET_CHANGED_KEY,
        page_context_key=_PAGE_CONTEXT_KEY,
        preset_select_key=_PRESET_SELECT_KEY,
        preset_key=_PRESET_KEY,
        formula_key=_FORMULA_KEY,
        ui=OPT_UI_APPLY,
        defaults_factory=lambda c: opt_ui_defaults(
            wl_from=c.recommended_wl_from,
            wl_to=c.recommended_wl_to,
            formula=c.initial_formula,
        ),
    )()


def _params_stale(
    session: FreehandSession,
    formula: str,
    *,
    wl_from: float,
    wl_to: float,
    angle_deg: float,
    polarization: str,
) -> bool:
    if not session.built:
        return False
    if formula.strip() != session.working_formula:
        return True
    if abs(angle_deg - session.angle_deg) > 1e-6:
        return True
    if polarization != session.polarization:
        return True
    if abs(wl_from - session.wl_from) > 1e-9 or abs(wl_to - session.wl_to) > 1e-9:
        return True
    return False


def _sync_view_domain(session: FreehandSession, event: dict[str, Any]) -> None:
    view_domain = event.get("viewDomain")
    if view_domain:
        session.view_domain = view_domain


def _bump_render_gen() -> None:
    st.session_state[_RENDER_GEN_KEY] = int(st.session_state.get(_RENDER_GEN_KEY, 0)) + 1


def _handle_component_event(session: FreehandSession, event: dict[str, Any]) -> bool:
    """Return True if optimization should run."""
    etype = event.get("type")
    _sync_view_domain(session, event)
    if etype == "activeMetric":
        session.active_metric = str(event.get("activeMetric", "R"))
        return False
    if etype == "viewChange":
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
                session.target[m] = clamp_freehand_target_array(np.asarray(target[m], dtype=float))
            if touched.get(m):
                session.touched[m] = True
        if metric in METRICS and edit_indices.get(metric):
            session.edit_wl_indices[metric] |= {int(i) for i in edit_indices[metric]}
        return bool(event.get("triggerOptimize")) and any(session.touched.values())
    return False


def _run_freehand_optimize(
    session: FreehandSession,
    db: Dict[str, Any],
    *,
    film_indices: list[int],
    thicknesses_um: list[float],
) -> str | None:
    """Run L-BFGS-B synchronously. Return success message, or None on failure."""
    session.optimizing = True
    try:
        with st.spinner("L-BFGS-B 优化中…"):
            pre_current = {
                k: np.asarray(v, dtype=float).copy() for k, v in session.current.items()
            }
            validate_freehand_targets(session.touched, session.target)
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
                film_indices=film_indices,
                thicknesses_um=thicknesses_um,
                layer_range_pct=session.layer_range_pct,
            )
            opt_formula, merit_history, current, merit_initial = run_freehand_optimize(cfg, db)
            session.apply_optimization_result(
                formula=opt_formula,
                current=current,
                merit_history=merit_history,
                merit_initial=merit_initial,
                pre_optimize_current=pre_current,
            )
            st.session_state["fs_opt_reset_component"] = True
            _bump_render_gen()
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
    *,
    film_indices: list[int],
    thicknesses_um: list[float],
) -> bool:
    """Handle component event; run optimization inline when triggered. Return True if rerun needed."""
    should_optimize = _handle_component_event(session, event)
    if should_optimize:
        message = _run_freehand_optimize(
            session,
            db,
            film_indices=film_indices,
            thicknesses_um=thicknesses_um,
        )
        if message is not None:
            st.session_state[_OPT_SUCCESS_KEY] = message
            return True
        return False
    return False


def render_page(
    *,
    context: PageContext,
    materials_db: Optional[Dict[str, Any]] = None,
) -> None:
    db, _, preset_ids, preset_labels, default_wl_from, default_wl_to = bootstrap_filmstack_page(
        page_title="Freehand 局部优化",
        inject_styles=inject_filmstack_opt_styles,
        context=context,
        keys=_SESSION_KEYS,
        materials_db=materials_db,
        ui=OPT_UI_APPLY,
        ui_defaults=opt_ui_defaults(
            wl_from=context.recommended_wl_from,
            wl_to=context.recommended_wl_to,
            formula=context.initial_formula,
        ),
    )
    session = _session()
    _hydrate_widgets_from_session(session)

    st.markdown('<h1 class="fs-opt-title">Freehand 局部优化</h1>', unsafe_allow_html=True)

    panel_head(
        "多层膜构建指令",
        css_prefix="fs-opt",
        help_text=FORMULA_HELP_TEXT,
        help_url=FORMULA_DOCS_URL,
    )
    st.markdown('<div class="fs-opt-input-row-marker"></div>', unsafe_allow_html=True)
    formula_col, params_col = st.columns(_INPUT_ROW_COLS, gap="small")
    with formula_col:
        st.markdown('<div class="fs-opt-formula-area-marker"></div>', unsafe_allow_html=True)
        formula = st.text_area(
            "多层膜构建指令",
            height=OPT_FORMULA_STACK_HEIGHT_PX,
            label_visibility="collapsed",
            key=_FORMULA_KEY,
        )
    with params_col:
        st.markdown('<div class="fs-opt-params-stack-marker"></div>', unsafe_allow_html=True)
        preset_select_row(
            preset_options=range(len(preset_ids)),
            preset_format_func=lambda i: preset_labels[preset_ids[i]],
            preset_key=_PRESET_SELECT_KEY,
            preset_on_change=_on_preset_change,
            css_prefix="fs-opt",
        )
        wl_from, wl_to = range_inputs(
            WL_RANGE_LABEL,
            css_prefix="fs-opt",
            key_from=_WL_FROM_KEY,
            key_to=_WL_TO_KEY,
            default_from=default_wl_from,
            default_to=default_wl_to,
            fmt="%.4f",
        )
        angle_fix, polarization = angle_polarization_inputs(
            "入射角 θ (°)/偏振",
            css_prefix="fs-opt",
            angle_key=_ANGLE_KEY,
            angle_default=DEFAULT_FIXED_ANGLE,
            angle_fmt="%.2f",
            polarization_key=_POLARIZATION_KEY,
            polarization_on_change=_on_polarization_change,
        )
        sim_clicked = st.button("仿真", key="fs_opt_build", width="stretch")
        params_stack_spacer(css_prefix="fs-opt")

    angle_fix = _clamp_angle(angle_fix)
    polarization = str(polarization).upper()

    just_built = False
    if sim_clicked and formula.strip():
        try:
            materials, thicknesses_um, layers = resolve_stack_with_layers(formula.strip(), db)
            if filmstack_visualizer.layers_has_incoherent(layers):
                st.warning(
                    "检测到非相干膜层（厚度带 * 后缀）。"
                    "当前优化器的梯度仍按相干模型计算，结果可能不准确；非相干优化待后续再开发。"
                )
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
                film_indices=film_layer_indices(len(thicknesses_um)),
                default_range_pct=get_freehand_default_thickness_range_pct(),
            )
            st.session_state["fs_opt_reset_component"] = True
            _bump_render_gen()
            st.session_state.pop(_LAYER_TABLE_KEY, None)
            st.session_state.pop(_POLARIZATION_CHANGED_KEY, None)
            st.session_state.pop(_PRESET_CHANGED_KEY, None)
            just_built = True
        except Exception as exc:
            st.error(f"仿真失败: {exc}")

    show_rebuild_prompt(
        has_built=session.built,
        polarization_changed=bool(st.session_state.pop(_POLARIZATION_CHANGED_KEY, False)),
        preset_changed=bool(st.session_state.pop(_PRESET_CHANGED_KEY, False)),
        params_stale=_params_stale(
            session,
            formula,
            wl_from=wl_from,
            wl_to=wl_to,
            angle_deg=angle_fix,
            polarization=polarization,
        ),
    )

    if session.built:
        materials: list[Any] = []
        thicknesses_um: list[float] = []
        film_indices: list[int] = []
        stack_resolved = False
        try:
            materials, thicknesses_um = _resolve_stack_cached(session.working_formula, db)
            film_indices = film_layer_indices(len(thicknesses_um))
            stack_resolved = True
            layer_table = stack_table_rows(
                materials,
                thicknesses_um,
                layer_range_pct=session.layer_range_pct,
                film_indices=film_indices,
            )
            st.markdown('<div class="fs-opt-layer-table-marker"></div>', unsafe_allow_html=True)
            expected_rows = len(materials)
            cached_table = st.session_state.get(_LAYER_TABLE_KEY)
            if hasattr(cached_table, "__len__") and len(cached_table) != expected_rows:
                st.session_state.pop(_LAYER_TABLE_KEY, None)
            edited_table = st.data_editor(
                layer_table,
                width="stretch",
                hide_index=False,
                disabled=["材料", "厚度 (μm)"],
                column_config={
                    "_idx": None,
                    "厚度变化范围 (%)": st.column_config.NumberColumn(
                        "厚度变化范围 (%)",
                        min_value=0,
                        max_value=100,
                        step=1,
                        format="%.0f",
                        alignment="left",
                    ),
                },
                key=_LAYER_TABLE_KEY,
            )
            if "_idx" in edited_table.columns:
                valid_idx = edited_table["_idx"].isin(range(len(materials)))
                edited_table = edited_table.loc[valid_idx]
            session.layer_range_pct.update(
                sync_layer_range_pct_from_table(edited_table, film_indices)
            )
        except Exception as exc:
            st.warning(f"层表解析失败: {exc}")

        if not stack_resolved:
            if session.optimizing:
                session.optimizing = False
        elif session.optimizing and not any(session.touched.values()):
            session.optimizing = False

        if stack_resolved:
            panel_head(
                "R / T / A vs λ（Freehand）",
                css_prefix="fs-opt",
                help_text=FREEHAND_CHART_HELP_TEXT,
            )

            component_args = session.to_component_args()
            component_args["renderGen"] = int(st.session_state.get(_RENDER_GEN_KEY, 0))
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
                    etype = event.get("type")
                    if etype in _VIEW_UI_EVENTS:
                        _handle_component_event(session, event)
                        _bump_render_gen()
                        st.rerun()
                    else:
                        needs_rerun = _process_component_event(
                            session,
                            event,
                            db,
                            film_indices=film_indices,
                            thicknesses_um=thicknesses_um,
                        )

            if needs_rerun:
                st.rerun()

            opt_success = st.session_state.pop(_OPT_SUCCESS_KEY, None)
            if opt_success:
                st.success(opt_success)

            panel_head(
                "优化后膜系指令",
                css_prefix="fs-opt",
                help_text=OPTIMIZED_FORMULA_HELP_TEXT,
            )
            if session.last_optimized_formula:
                st.markdown(
                    '<div class="fs-opt-optimized-formula-marker"></div>',
                    unsafe_allow_html=True,
                )
                st.code(session.last_optimized_formula, language=None)
            else:
                st.caption("完成一次 Freehand 优化后将在此显示膜系指令。")
