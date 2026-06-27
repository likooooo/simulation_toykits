"""Filmstack Simulation Streamlit page."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import streamlit as st

import filmstack_visualizer
from filmstack_simulation.help_texts import (
    FORMULA_DOCS_URL,
    FORMULA_HELP_TEXT,
    WL_RANGE_LABEL,
)
from filmstack_simulation.page_shell import (
    FilmstackSessionKeys,
    PageContext,
    bootstrap_filmstack_page,
    ensure_filmstack_session_defaults,
)
from filmstack_simulation.page_styles import inject_filmstack_sim_styles
from filmstack_simulation.plots import show_figure, show_plotly_figure
from filmstack_simulation.presets import PresetCatalog
from filmstack_simulation.simulation import (
    compute_polarized_curve_at_angle,
    compute_polarized_curve_at_wavelength,
    compute_spectral_map_2d,
    resolve_stack_with_layers,
)
from filmstack_simulation.page_widgets import (
    on_preset_change,
    preset_polarization_row,
    resolve_stack_cached,
    POLARIZATION_LABELS,
    panel_head,
    range_inputs,
    show_rebuild_prompt,
    target_wl_ang_inputs,
    FORMULA_STACK_HEIGHT_PX,
)

FORMULA_KEY = "fs_sim_formula"
PRESET_KEY = "fs_sim_preset"
PRESET_SELECT_KEY = "fs_sim_preset_select"
STACK_BUILT_FORMULA_KEY = "fs_sim_stack_built_formula"
STACK_RESOLVED_KEY = "fs_sim_stack_resolved"
SLICE_FIGS_KEY = "fs_sim_slice_figs"
MAP2D_FIG_KEY = "fs_sim_map2d_fig"
MAP2D_NK_FIG_KEY = "fs_sim_map2d_nk_fig"
_POLARIZATION_KEY = "fs_sim_polarization"
_POLARIZATION_CHANGED_KEY = "fs_sim_polarization_changed"
_PRESET_CHANGED_KEY = "fs_sim_preset_changed"
_BUILD_SNAPSHOT_KEY = "fs_sim_build_snapshot"

DEFAULT_ANG_FROM = 0.0
DEFAULT_ANG_TO = 60.0
DEFAULT_TARGET_WL = 0.55
DEFAULT_TARGET_ANG = 0.0

_INPUT_ROW_COLS = [1.5, 1]

_PAGE_CONTEXT_KEY = "_fs_sim_page_context"
_SESSION_KEYS = FilmstackSessionKeys(
    formula_key=FORMULA_KEY,
    preset_key=PRESET_KEY,
    preset_select_key=PRESET_SELECT_KEY,
    polarization_key=_POLARIZATION_KEY,
    page_context_key=_PAGE_CONTEXT_KEY,
)


def ensure_session_defaults(
    materials_db: Optional[Dict[str, Any]] = None,
    *,
    preset_catalog: PresetCatalog,
    initial_preset_id: str,
    initial_formula: str,
    sim_wl_from: float | None = None,
    sim_wl_to: float | None = None,
) -> None:
    ensure_filmstack_session_defaults(
        materials_db,
        keys=_SESSION_KEYS,
        preset_catalog=preset_catalog,
        initial_preset_id=initial_preset_id,
        initial_formula=initial_formula,
        sim_wl_from=sim_wl_from,
        sim_wl_to=sim_wl_to,
    )


def clear_slice_figs() -> None:
    st.session_state.pop(SLICE_FIGS_KEY, None)


def _clear_map_figs() -> None:
    st.session_state.pop(MAP2D_FIG_KEY, None)
    st.session_state.pop(MAP2D_NK_FIG_KEY, None)


def _on_polarization_change() -> None:
    clear_slice_figs()
    _clear_map_figs()
    st.session_state[_POLARIZATION_CHANGED_KEY] = True


def _resolve_stack_cached(
    formula: str, db: Dict[str, Any]
) -> tuple[list[Any], list[float], list[Any]]:
    return resolve_stack_cached(
        formula,
        db,
        cache_key=STACK_RESOLVED_KEY,
        resolve=resolve_stack_with_layers,
    )


def _layers_from_formula(formula: str, db: Dict[str, Any]) -> list[Any]:
    _, _, layers = _resolve_stack_cached(formula, db)
    return layers


GetMaterialsDb = Callable[[], Dict[str, Any]]


def _sim_snapshot_stale(
    snapshot: dict[str, Any] | None,
    *,
    formula: str,
    wl_from: float,
    wl_to: float,
    ang_from: float,
    ang_to: float,
    target_wl: float,
    target_ang: float,
    polarization: str,
) -> bool:
    if snapshot is None:
        return False
    if formula.strip() != str(snapshot.get("formula", "")).strip():
        return True
    if polarization != snapshot.get("polarization"):
        return True
    if abs(wl_from - float(snapshot["wl_from"])) > 1e-9 or abs(wl_to - float(snapshot["wl_to"])) > 1e-9:
        return True
    if abs(ang_from - float(snapshot["ang_from"])) > 1e-6 or abs(ang_to - float(snapshot["ang_to"])) > 1e-6:
        return True
    if abs(target_wl - float(snapshot["target_wl"])) > 1e-9:
        return True
    if abs(target_ang - float(snapshot["target_ang"])) > 1e-6:
        return True
    return False


def _run_full_simulation(
    formula: str,
    db: Dict[str, Any],
    *,
    wl_from: float,
    wl_to: float,
    ang_from: float,
    ang_to: float,
    polarization: str,
    target_wl: float,
    target_ang: float,
) -> None:
    plot_formula = formula.strip()
    materials, thicknesses_um, layers = _resolve_stack_cached(plot_formula, db)
    st.session_state[STACK_BUILT_FORMULA_KEY] = plot_formula

    with st.spinner("计算中…"):
        cache = compute_spectral_map_2d(
            materials,
            thicknesses_um,
            wl_from,
            wl_to,
            ang_from,
            ang_to,
            polarization=polarization,
            layers=layers,
        )
        at_wl = compute_polarized_curve_at_wavelength(
            materials,
            thicknesses_um,
            target_wl,
            ang_from,
            ang_to,
            polarization=polarization,
        )
        at_ang = compute_polarized_curve_at_angle(
            materials,
            thicknesses_um,
            target_ang,
            wl_from,
            wl_to,
            polarization=polarization,
        )

    nk_fig = filmstack_visualizer.plot_filmstack_material_nk_1x2(
        materials, cache["wavelength_um"]
    )
    map_fig = filmstack_visualizer.plot_filmstack_rt_psi_delta_map_2x2(
        cache["wavelength_um"],
        cache["angle_deg"],
        cache["R"],
        cache["T"],
        cache["Psi"],
        cache["Delta"],
    )
    fig_wl = filmstack_visualizer.plot_filmstack_rt_psi_delta_slice_2x2(
        at_wl["x"],
        at_wl["R"],
        at_wl["T"],
        at_wl["Psi"],
        at_wl["Delta"],
        xlabel="Angle (°)",
        title_prefix=f"@ {target_wl:.4f} μm — ",
    )
    fig_ang = filmstack_visualizer.plot_filmstack_rt_psi_delta_slice_2x2(
        at_ang["x"],
        at_ang["R"],
        at_ang["T"],
        at_ang["Psi"],
        at_ang["Delta"],
        xlabel="Wavelength (μm)",
        title_prefix=f"@ {target_ang:.1f}° — ",
    )
    st.session_state[MAP2D_NK_FIG_KEY] = nk_fig
    st.session_state[MAP2D_FIG_KEY] = map_fig
    st.session_state[SLICE_FIGS_KEY] = (fig_wl, fig_ang)
    st.session_state[_BUILD_SNAPSHOT_KEY] = {
        "formula": plot_formula,
        "wl_from": wl_from,
        "wl_to": wl_to,
        "ang_from": ang_from,
        "ang_to": ang_to,
        "target_wl": target_wl,
        "target_ang": target_ang,
        "polarization": polarization,
    }
    st.session_state.pop(_POLARIZATION_CHANGED_KEY, None)
    st.session_state.pop(_PRESET_CHANGED_KEY, None)


def _on_preset_change() -> None:
    st.session_state[_PRESET_CHANGED_KEY] = True
    ctx = st.session_state.get(_PAGE_CONTEXT_KEY)
    if ctx is None:
        return
    on_preset_change(
        preset_select_key=PRESET_SELECT_KEY,
        preset_key=PRESET_KEY,
        formula_key=FORMULA_KEY,
        page_context_key=_PAGE_CONTEXT_KEY,
        preset_ids=ctx.preset_catalog.preset_ids,
    )


def render_page(
    *,
    context: PageContext,
    materials_db: Optional[Dict[str, Any]] = None,
) -> None:
    """Render the Filmstack Simulation page."""

    db, _, preset_ids, preset_labels, default_wl_from, default_wl_to = bootstrap_filmstack_page(
        page_title="多层膜仿真",
        inject_styles=inject_filmstack_sim_styles,
        context=context,
        keys=_SESSION_KEYS,
        materials_db=materials_db,
    )

    st.markdown('<h1 class="fs-sim-title">多层膜仿真</h1>', unsafe_allow_html=True)

    panel_head(
        "多层膜构建指令",
        css_prefix="fs-sim",
        help_text=FORMULA_HELP_TEXT,
        help_url=FORMULA_DOCS_URL,
    )
    st.markdown('<div class="fs-sim-input-row-marker"></div>', unsafe_allow_html=True)
    formula_col, params_col = st.columns(_INPUT_ROW_COLS, gap="small")
    with formula_col:
        st.markdown('<div class="fs-sim-formula-area-marker"></div>', unsafe_allow_html=True)
        formula = st.text_area(
            "多层膜构建指令",
            height=FORMULA_STACK_HEIGHT_PX,
            help=FORMULA_HELP_TEXT,
            label_visibility="collapsed",
            key=FORMULA_KEY,
        )
    with params_col:
        st.markdown('<div class="fs-sim-params-stack-marker"></div>', unsafe_allow_html=True)
        polarization = preset_polarization_row(
            preset_options=range(len(preset_ids)),
            preset_format_func=lambda i: preset_labels[preset_ids[i]],
            preset_key=PRESET_SELECT_KEY,
            preset_on_change=_on_preset_change,
            polarization_key=_POLARIZATION_KEY,
            polarization_on_change=_on_polarization_change,
            css_prefix="fs-sim",
        )
        wl_from, wl_to = range_inputs(
            WL_RANGE_LABEL,
            css_prefix="fs-sim",
            key_from="fs_sim_wl_from",
            key_to="fs_sim_wl_to",
            default_from=default_wl_from,
            default_to=default_wl_to,
            fmt="%.4f",
        )
        ang_from, ang_to = range_inputs(
            "仿真角度范围\u00a0(°)",
            css_prefix="fs-sim",
            key_from="fs_sim_ang_from",
            key_to="fs_sim_ang_to",
            default_from=DEFAULT_ANG_FROM,
            default_to=DEFAULT_ANG_TO,
            fmt="%.2f",
        )
        target_wl, target_ang = target_wl_ang_inputs(
            "目标波长/角度",
            css_prefix="fs-sim",
            wl_key="fs_sim_target_wl",
            ang_key="fs_sim_target_ang",
            wl_default=DEFAULT_TARGET_WL,
            ang_default=DEFAULT_TARGET_ANG,
            wl_fmt="%.4f",
            ang_fmt="%.2f",
        )
        sim_clicked = st.button("仿真", key="fs_sim_build", width="stretch")

    polarization = str(polarization).upper()
    pol_label = POLARIZATION_LABELS.get(polarization, polarization)

    if sim_clicked and formula.strip():
        try:
            _run_full_simulation(
                formula,
                db,
                wl_from=wl_from,
                wl_to=wl_to,
                ang_from=ang_from,
                ang_to=ang_to,
                polarization=polarization,
                target_wl=target_wl,
                target_ang=target_ang,
            )
        except Exception as exc:
            st.error(f"仿真失败: {exc}")
            import traceback

            st.code(traceback.format_exc())

    built_formula = st.session_state.get(STACK_BUILT_FORMULA_KEY)
    build_snapshot = st.session_state.get(_BUILD_SNAPSHOT_KEY)
    show_rebuild_prompt(
        has_built=bool(built_formula),
        polarization_changed=bool(st.session_state.pop(_POLARIZATION_CHANGED_KEY, False)),
        preset_changed=bool(st.session_state.pop(_PRESET_CHANGED_KEY, False)),
        params_stale=_sim_snapshot_stale(
            build_snapshot,
            formula=formula,
            wl_from=wl_from,
            wl_to=wl_to,
            ang_from=ang_from,
            ang_to=ang_to,
            target_wl=target_wl,
            target_ang=target_ang,
            polarization=polarization,
        ),
    )

    if built_formula:
        try:
            layers = _layers_from_formula(built_formula, db)
            fig = filmstack_visualizer.plot_filmstack(layers, layer_label_mode="legend", show=False)
            show_figure(fig)
        except Exception as exc:
            st.error(f"膜系展示失败: {exc}")

    if MAP2D_NK_FIG_KEY in st.session_state:
        st.markdown(
            '<div class="fs-sim-chart-title">材料 n / k 曲线</div>',
            unsafe_allow_html=True,
        )
        show_plotly_figure(st.session_state[MAP2D_NK_FIG_KEY], key="fs_sim_nk_chart")

    if MAP2D_FIG_KEY in st.session_state:
        st.markdown(
            f'<div class="fs-sim-chart-title">Spectral map ({pol_label})</div>',
            unsafe_allow_html=True,
        )
        st.caption("Ψ/Δ 由 s/p 反射系数比计算，与上方偏振选择无关。")
        show_figure(st.session_state[MAP2D_FIG_KEY])

    if SLICE_FIGS_KEY in st.session_state:
        fig_wl, fig_ang = st.session_state[SLICE_FIGS_KEY]
        st.markdown(
            '<div class="fs-sim-chart-title">切片 @ 目标角度</div>',
            unsafe_allow_html=True,
        )
        show_figure(fig_ang)
        st.markdown(
            '<div class="fs-sim-chart-title">切片 @ 目标波长</div>',
            unsafe_allow_html=True,
        )
        show_figure(fig_wl)
