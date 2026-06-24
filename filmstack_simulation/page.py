"""Filmstack Simulation Streamlit page."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import streamlit as st

import filmstack_visualizer
from filmstack_simulation.page_styles import inject_filmstack_sim_styles
from filmstack_simulation.plots import show_figure, show_plotly_figure
from filmstack_simulation.presets import (
    DEFAULT_PRESET_ID,
    PRESET_IDS,
    PRESET_LABELS,
    build_formula_for_preset,
    get_wl_mid_um,
)
from filmstack_simulation.simulation import (
    compute_polarized_curve_at_angle,
    compute_polarized_curve_at_wavelength,
    compute_spectral_map_2d,
    resolve_stack_with_layers,
)
from filmstack_simulation.materials import RECOMMENDED_SIM_WL_FROM_UM, RECOMMENDED_SIM_WL_TO_UM
from filmstack_simulation.page_widgets import (
    init_formula_default,
    init_preset_select,
    on_preset_change,
    resolve_stack_cached,
    DEFAULT_POLARIZATION,
    POLARIZATION_LABELS,
    panel_head,
    range_inputs,
    range_inputs_with_polarization,
    single_input,
)

_TOKENS_PATH = Path(__file__).resolve().parent / "design_tokens.css"

def inject_global_styles() -> None:
    """Inject page CSS every rerun — Streamlit discards prior markdown on interaction."""
    inject_filmstack_sim_styles(_TOKENS_PATH)



FORMULA_KEY = "fs_sim_formula"
PRESET_KEY = "fs_sim_preset"
PRESET_SELECT_KEY = "fs_sim_preset_select"
STACK_BUILT_FORMULA_KEY = "fs_sim_stack_built_formula"
STACK_RESOLVED_KEY = "fs_sim_stack_resolved"
SLICE_FIGS_KEY = "fs_sim_slice_figs"
MAP2D_FIG_KEY = "fs_sim_map2d_fig"
MAP2D_NK_FIG_KEY = "fs_sim_map2d_nk_fig"
_POLARIZATION_KEY = "fs_sim_polarization"

DEFAULT_WL_FROM = RECOMMENDED_SIM_WL_FROM_UM
DEFAULT_WL_TO = RECOMMENDED_SIM_WL_TO_UM
DEFAULT_ANG_FROM = 0.0
DEFAULT_ANG_TO = 60.0
DEFAULT_TARGET_WL = 0.55
DEFAULT_TARGET_ANG = 0.0


def ensure_session_defaults(
    materials_db: Optional[Dict[str, Any]] = None,
    *,
    sim_wl_from: float | None = None,
    sim_wl_to: float | None = None,
) -> None:
    init_preset_select(
        preset_key=PRESET_KEY,
        preset_select_key=PRESET_SELECT_KEY,
        preset_ids=PRESET_IDS,
        default_preset_id=DEFAULT_PRESET_ID,
    )
    if FORMULA_KEY not in st.session_state:
        if materials_db:
            wl_mid = get_wl_mid_um(sim_wl_from, sim_wl_to)
            default_formula = build_formula_for_preset(
                DEFAULT_PRESET_ID,
                materials_db,
                wl_mid,
            )
        else:
            default_formula = ""
        init_formula_default(
            formula_key=FORMULA_KEY,
            materials_db=materials_db,
            default_formula=default_formula,
        )
    if _POLARIZATION_KEY not in st.session_state:
        st.session_state[_POLARIZATION_KEY] = DEFAULT_POLARIZATION


def clear_slice_figs() -> None:
    st.session_state.pop(SLICE_FIGS_KEY, None)


def _clear_map_figs() -> None:
    st.session_state.pop(MAP2D_FIG_KEY, None)
    st.session_state.pop(MAP2D_NK_FIG_KEY, None)


def _on_polarization_change() -> None:
    clear_slice_figs()
    _clear_map_figs()
    st.session_state["fs_sim_polarization_changed"] = True


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


@dataclass(frozen=True)
class PageContext:
    get_materials_db: GetMaterialsDb
    sim_wl_from: float | None = None
    sim_wl_to: float | None = None


_DOCS_URL = "https://github.com/likooooo/simulation_toykits/blob/main/docs/filmstack_formula_usage.md"
_BUILD_ROW_COLS = [5, 1]
_BUILD_FORMULA_HEIGHT = 132
_PAGE_CONTEXT_KEY = "_fs_sim_page_context"


def _require_built_formula(built_formula: str | None, formula: str) -> str | None:
    """Return the built formula for plotting, or None after showing a warning."""
    if not built_formula:
        st.warning("请先输入多层膜构建指令并点击「构建」")
        return None
    if formula.strip() != built_formula:
        st.warning("公式已修改，请先点击「构建」再绘制")
        return None
    return built_formula


def _action_button(label: str, key: str) -> bool:
    st.markdown('<div class="fs-sim-action-row">', unsafe_allow_html=True)
    _, btn_col = st.columns([7, 1])
    with btn_col:
        clicked = st.button(label, key=key, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)
    return clicked


def _on_preset_change() -> None:
    on_preset_change(
        preset_select_key=PRESET_SELECT_KEY,
        preset_key=PRESET_KEY,
        formula_key=FORMULA_KEY,
        page_context_key=_PAGE_CONTEXT_KEY,
        preset_ids=PRESET_IDS,
    )


def render_page(
    *,
    context: PageContext,
    materials_db: Optional[Dict[str, Any]] = None,
) -> None:
    """Render the Filmstack Simulation page."""
    st.set_page_config(page_title="多层膜仿真", layout="wide")
    inject_global_styles()
    st.session_state[_PAGE_CONTEXT_KEY] = context
    db = materials_db if materials_db is not None else context.get_materials_db()
    ensure_session_defaults(
        db if db else None,
        sim_wl_from=context.sim_wl_from,
        sim_wl_to=context.sim_wl_to,
    )

    st.markdown('<div class="fs-sim-page-marker"></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="fs-sim-title">多层膜仿真</h1>', unsafe_allow_html=True)

    # --- Section 1: build filmstack ---
    panel_head("多层膜构建指令", css_prefix="fs-sim", help_url=_DOCS_URL)
    st.markdown('<div class="fs-sim-build-row-marker"></div>', unsafe_allow_html=True)
    formula_col, action_col = st.columns(_BUILD_ROW_COLS, gap="small")
    with formula_col:
        formula = st.text_area(
            "多层膜构建指令",
            height=_BUILD_FORMULA_HEIGHT,
            help="格式：Material thickness_um [n k]；(…)^N 周期。不在材料库的材料必须写 n k。",
            label_visibility="collapsed",
            key=FORMULA_KEY,
        )
    with action_col:
        st.markdown('<div class="fs-sim-build-right-marker"></div>', unsafe_allow_html=True)
        st.selectbox(
            "预设膜系",
            range(len(PRESET_IDS)),
            format_func=lambda i: PRESET_LABELS[PRESET_IDS[i]],
            key=PRESET_SELECT_KEY,
            on_change=_on_preset_change,
            label_visibility="collapsed",
        )
        build_clicked = st.button("构建", key="fs_sim_build", width="stretch")

    if build_clicked and formula.strip():
        try:
            _resolve_stack_cached(formula.strip(), db)
            st.session_state[STACK_BUILT_FORMULA_KEY] = formula.strip()
            clear_slice_figs()
            _clear_map_figs()
        except Exception as exc:
            st.error(f"构建失败: {exc}")

    built_formula = st.session_state.get(STACK_BUILT_FORMULA_KEY)
    if built_formula:
        try:
            layers = _layers_from_formula(built_formula, db)
            fig = filmstack_visualizer.plot_filmstack(layers, layer_label_mode="legend", show=False)
            show_figure(fig)
        except Exception as exc:
            st.error(f"膜系展示失败: {exc}")

    st.divider()

    # --- Section 2: 2D spectral map ---
    st.markdown(
        '<div class="fs-sim-section-label">二维仿真 (R / T / Ψ / Δ)</div>',
        unsafe_allow_html=True,
    )
    col_wl, _col_mid, col_ang = st.columns([1, 0.75, 1], gap="small")
    with col_wl:
        st.markdown('<div class="fs-sim-param-left"></div>', unsafe_allow_html=True)
        wl_from, wl_to = range_inputs(
            "仿真波长范围\u00a0(μm)",
            css_prefix="fs-sim",
            key_from="fs_sim_wl_from",
            key_to="fs_sim_wl_to",
            default_from=DEFAULT_WL_FROM,
            default_to=DEFAULT_WL_TO,
            fmt="%.4f",
        )
    with col_ang:
        st.markdown('<div class="fs-sim-param-right"></div>', unsafe_allow_html=True)
        ang_from, ang_to, polarization = range_inputs_with_polarization(
            "仿真角度范围\u00a0(°)",
            css_prefix="fs-sim",
            key_from="fs_sim_ang_from",
            key_to="fs_sim_ang_to",
            default_from=DEFAULT_ANG_FROM,
            default_to=DEFAULT_ANG_TO,
            fmt="%.2f",
            polarization_key=_POLARIZATION_KEY,
            on_polarization_change=_on_polarization_change,
        )
    polarization = str(polarization).upper()
    pol_label = POLARIZATION_LABELS.get(polarization, polarization)
    draw2d_clicked = _action_button("绘制", "fs_sim_draw_2d")

    if draw2d_clicked:
        plot_formula = _require_built_formula(built_formula, formula)
        if plot_formula:
            try:
                materials, thicknesses_um, layers = _resolve_stack_cached(plot_formula, db)
                with st.spinner("计算波长-角度二维图…"):
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
                    title=f"光谱图 ({pol_label})",
                )
                st.session_state[MAP2D_NK_FIG_KEY] = nk_fig
                st.session_state[MAP2D_FIG_KEY] = map_fig
                clear_slice_figs()
            except Exception as exc:
                st.error(f"绘制失败: {exc}")
                import traceback

                st.code(traceback.format_exc())

    if MAP2D_NK_FIG_KEY in st.session_state:
        st.markdown(
            '<div class="fs-sim-chart-title">材料 n / k 曲线</div>',
            unsafe_allow_html=True,
        )
        show_plotly_figure(st.session_state[MAP2D_NK_FIG_KEY], key="fs_sim_nk_chart")

    if MAP2D_FIG_KEY in st.session_state:
        st.markdown(
            f'<div class="fs-sim-chart-title">光谱图 ({pol_label})</div>',
            unsafe_allow_html=True,
        )
        st.caption("Ψ/Δ 由 s/p 反射系数比计算，与上方偏振选择无关。")
        show_figure(st.session_state[MAP2D_FIG_KEY])

    st.divider()

    if st.session_state.pop("fs_sim_polarization_changed", False):
        st.info("偏振已更改，请重新点击「绘制」。")

    # --- Section 3: 1D slices ---
    st.markdown(
        '<div class="fs-sim-section-label">一维切片</div>',
        unsafe_allow_html=True,
    )
    col_twl, _col_mid, col_tang = st.columns([1, 0.75, 1], gap="small")
    with col_twl:
        st.markdown('<div class="fs-sim-param-left"></div>', unsafe_allow_html=True)
        target_wl = single_input(
            "目标波长\u00a0(μm)",
            css_prefix="fs-sim",
            key="fs_sim_target_wl",
            default=DEFAULT_TARGET_WL,
            fmt="%.4f",
        )
    with col_tang:
        st.markdown('<div class="fs-sim-param-right"></div>', unsafe_allow_html=True)
        target_ang = single_input(
            "目标角度\u00a0(°)",
            css_prefix="fs-sim",
            key="fs_sim_target_ang",
            default=DEFAULT_TARGET_ANG,
            fmt="%.2f",
        )
    draw1d_clicked = _action_button("绘制", "fs_sim_draw_1d")

    if draw1d_clicked:
        plot_formula = _require_built_formula(built_formula, formula)
        if plot_formula:
            try:
                materials, thicknesses_um, _ = _resolve_stack_cached(plot_formula, db)
                with st.spinner("计算一维切片…"):
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
                st.session_state[SLICE_FIGS_KEY] = (fig_wl, fig_ang)
            except Exception as exc:
                st.error(f"切片绘制失败: {exc}")

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