"""Shared Streamlit widgets for filmstack simulation and optimization pages."""

from __future__ import annotations

from typing import Any, Callable, Dict, Literal, Optional, Sequence

import streamlit as st

from filmstack_simulation.component.panel_section_head import panel_section_head
from filmstack_simulation.help_texts import (
    PARAMS_STALE_INFO,
    POLARIZATION_STALE_INFO,
    PRESET_STALE_INFO,
)
from filmstack_simulation.presets import CUSTOM_PRESET_ID, PresetCatalog, build_formula_for_preset
from filmstack_simulation.template_types import (
    FilmstackSimParams,
    FilmstackTemplate,
    FilmstackUIDefaults,
    FilmstackUIApplySpec,
)

FILMSTACK_TOKENS_CSS_KEY = "_filmstack_tokens_css"

POLARIZATION_IDS = ("TE", "TM", "UNPOLARIZED")
POLARIZATION_LABELS = {"TE": "TE", "TM": "TM", "UNPOLARIZED": "Unpolarized"}
DEFAULT_POLARIZATION = "UNPOLARIZED"
DEFAULT_ANG_FROM = 0.0
DEFAULT_ANG_TO = 60.0
DEFAULT_TARGET_WL = 0.55
DEFAULT_TARGET_ANG = 0.0
DEFAULT_FIXED_ANGLE = 0.0

SIM_UI_APPLY = FilmstackUIApplySpec(
    wl_from_key="fs_sim_wl_from",
    wl_to_key="fs_sim_wl_to",
    polarization_key="fs_sim_polarization",
    ang_from_key="fs_sim_ang_from",
    ang_to_key="fs_sim_ang_to",
    target_wl_key="fs_sim_target_wl",
    target_ang_key="fs_sim_target_ang",
)

OPT_UI_APPLY = FilmstackUIApplySpec(
    wl_from_key="fs_opt_wl_from",
    wl_to_key="fs_opt_wl_to",
    polarization_key="fs_opt_polarization",
    fixed_angle_key="fs_opt_angle",
)


def filmstack_ui_defaults(
    *,
    wl_from: float,
    wl_to: float,
    formula: str = "",
    mode: Literal["sim", "opt"] = "sim",
) -> FilmstackUIDefaults:
    if mode == "opt":
        return FilmstackUIDefaults(
            wl_from=wl_from,
            wl_to=wl_to,
            polarization=DEFAULT_POLARIZATION,
            fixed_angle=DEFAULT_FIXED_ANGLE,
            formula=formula,
        )
    return FilmstackUIDefaults(
        wl_from=wl_from,
        wl_to=wl_to,
        polarization=DEFAULT_POLARIZATION,
        ang_from=DEFAULT_ANG_FROM,
        ang_to=DEFAULT_ANG_TO,
        target_wl=DEFAULT_TARGET_WL,
        target_ang=DEFAULT_TARGET_ANG,
        formula=formula,
    )


def sim_ui_defaults(*, wl_from: float, wl_to: float, formula: str = "") -> FilmstackUIDefaults:
    return filmstack_ui_defaults(wl_from=wl_from, wl_to=wl_to, formula=formula, mode="sim")


def opt_ui_defaults(*, wl_from: float, wl_to: float, formula: str = "") -> FilmstackUIDefaults:
    return filmstack_ui_defaults(wl_from=wl_from, wl_to=wl_to, formula=formula, mode="opt")

# Left formula text area height (px); fixed to align with param stack visually.
SIM_FORMULA_STACK_HEIGHT_PX = 350
OPT_FORMULA_STACK_HEIGHT_PX = 240
# Outer row: label | controls = 1:2. Dual controls nest 1:1 inside the controls column.
_LABEL_CONTROL_COLS = [1, 2]
_DUAL_CONTROL_COLS = [1, 1]
PRESET_SELECT_LABEL = "膜系配置"


def _format_float(value: float, fmt: str) -> str:
    """Printf-style formatting matching ``st.number_input(format=...)``."""
    return fmt % float(value)


def _panel_head_columns(label: str, *, css_prefix: str):
    """Return controls column after rendering a panel-head label."""
    title, c_controls = st.columns(_LABEL_CONTROL_COLS, gap="small")
    with title:
        st.markdown(
            f'<div class="{css_prefix}-panel-head">{label}</div>',
            unsafe_allow_html=True,
        )
    return c_controls


def _migrate_numeric_session_value(*, key: str, fmt: str) -> None:
    """``st.number_input`` stored floats; ``st.text_input`` needs strings."""
    if key not in st.session_state:
        return
    stored = st.session_state[key]
    if isinstance(stored, (int, float)):
        st.session_state[key] = _format_float(stored, fmt)


def _parse_float_text(raw: str, *, default: float) -> float:
    text = str(raw).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _float_text_input(
    label: str,
    *,
    key: str,
    default: float,
    fmt: str,
) -> float:
    _migrate_numeric_session_value(key=key, fmt=fmt)
    kwargs: dict[str, str] = {}
    if key not in st.session_state:
        kwargs["value"] = _format_float(default, fmt)
    raw = st.text_input(
        label,
        key=key,
        label_visibility="collapsed",
        **kwargs,
    )
    return _parse_float_text(raw, default=default)


def show_rebuild_prompt(
    *,
    has_built: bool,
    polarization_changed: bool,
    preset_changed: bool,
    params_stale: bool,
) -> None:
    """Show a consistent rebuild hint after inputs drift from the last build."""
    if not has_built:
        return
    if polarization_changed:
        st.info(POLARIZATION_STALE_INFO)
    elif preset_changed:
        st.info(PRESET_STALE_INFO)
    elif params_stale:
        st.info(PARAMS_STALE_INFO)


def panel_head(
    label: str,
    *,
    css_prefix: str,
    help_text: str | None = None,
    help_url: str | None = None,
    help_url_label: str = "— 使用说明",
    align: Literal["left", "right"] = "left",
    key: str | None = None,
) -> None:
    panel_section_head(
        label,
        help_text=help_text,
        help_url=help_url,
        help_url_label=help_url_label,
        align=align,
        key=key,
        css_prefix=css_prefix,
        tokens_css=st.session_state.get(FILMSTACK_TOKENS_CSS_KEY, ""),
    )


def range_inputs(
    label: str,
    *,
    css_prefix: str,
    key_from: str,
    key_to: str,
    default_from: float,
    default_to: float,
    fmt: str,
) -> tuple[float, float]:
    c_controls = _panel_head_columns(label, css_prefix=css_prefix)
    with c_controls:
        c_from, c_to = st.columns(_DUAL_CONTROL_COLS, gap="small")
        with c_from:
            val_from = _float_text_input(
                "from",
                key=key_from,
                default=default_from,
                fmt=fmt,
            )
        with c_to:
            val_to = _float_text_input(
                "to",
                key=key_to,
                default=default_to,
                fmt=fmt,
            )
    return val_from, val_to


def target_wl_ang_inputs(
    label: str,
    *,
    css_prefix: str,
    wl_key: str,
    ang_key: str,
    wl_default: float,
    ang_default: float,
    wl_fmt: str,
    ang_fmt: str,
) -> tuple[float, float]:
    c_controls = _panel_head_columns(label, css_prefix=css_prefix)
    with c_controls:
        c_wl, c_ang = st.columns(_DUAL_CONTROL_COLS, gap="small")
        with c_wl:
            target_wl = _float_text_input(
                "target_wl",
                key=wl_key,
                default=wl_default,
                fmt=wl_fmt,
            )
        with c_ang:
            target_ang = _float_text_input(
                "target_ang",
                key=ang_key,
                default=ang_default,
                fmt=ang_fmt,
            )
    return target_wl, target_ang


def polarization_select(
    *,
    key: str,
    on_change: Callable[[], None] | None = None,
) -> str:
    kwargs: dict[str, Any] = {
        "label": "偏振",
        "options": POLARIZATION_IDS,
        "format_func": lambda p: POLARIZATION_LABELS[p],
        "key": key,
        "label_visibility": "collapsed",
    }
    if on_change is not None:
        kwargs["on_change"] = on_change
    return st.selectbox(**kwargs)


def params_stack_spacer(*, css_prefix: str) -> None:
    """Blank param row so opt params stack height matches the simulation page."""
    st.markdown(
        f'<div class="{css_prefix}-params-spacer-marker"></div>',
        unsafe_allow_html=True,
    )
    title, c_controls = st.columns(_LABEL_CONTROL_COLS, gap="small")
    with title:
        st.empty()
    with c_controls:
        c_from, c_to = st.columns(_DUAL_CONTROL_COLS, gap="small")
        with c_from:
            st.empty()
        with c_to:
            st.empty()


def preset_select_row(
    *,
    preset_options: Sequence[Any],
    preset_format_func: Callable[[Any], str],
    preset_key: str,
    preset_on_change: Callable[[], None] | None,
    css_prefix: str,
) -> None:
    st.markdown(
        f'<div class="{css_prefix}-preset-row-marker"></div>',
        unsafe_allow_html=True,
    )
    c_preset = _panel_head_columns(PRESET_SELECT_LABEL, css_prefix=css_prefix)
    with c_preset:
        preset_kwargs: dict[str, Any] = {
            "label": PRESET_SELECT_LABEL,
            "options": preset_options,
            "format_func": preset_format_func,
            "key": preset_key,
            "label_visibility": "collapsed",
        }
        if preset_on_change is not None:
            preset_kwargs["on_change"] = preset_on_change
        st.selectbox(**preset_kwargs)


def polarization_row(
    *,
    polarization_key: str,
    polarization_on_change: Callable[[], None] | None = None,
    css_prefix: str,
) -> str:
    st.markdown(
        f'<div class="{css_prefix}-pol-row-marker"></div>',
        unsafe_allow_html=True,
    )
    c_pol = _panel_head_columns("偏振", css_prefix=css_prefix)
    with c_pol:
        return polarization_select(
            key=polarization_key,
            on_change=polarization_on_change,
        )


def angle_polarization_inputs(
    label: str,
    *,
    css_prefix: str,
    angle_key: str,
    angle_default: float,
    angle_fmt: str,
    polarization_key: str,
    polarization_on_change: Callable[[], None] | None = None,
) -> tuple[float, str]:
    c_controls = _panel_head_columns(label, css_prefix=css_prefix)
    with c_controls:
        c_angle, c_pol = st.columns(_DUAL_CONTROL_COLS, gap="small")
        with c_angle:
            angle = _float_text_input(
                label,
                key=angle_key,
                default=angle_default,
                fmt=angle_fmt,
            )
        with c_pol:
            polarization = polarization_select(
                key=polarization_key,
                on_change=polarization_on_change,
            )
    return angle, polarization


def _apply_sim_param(
    key: str | None,
    value: float | str | None,
    *,
    default: float | str,
    fmt: str | None = None,
) -> None:
    if key is None:
        return
    effective = value if value is not None else default
    if fmt is not None and isinstance(effective, (int, float)):
        st.session_state[key] = _format_float(float(effective), fmt)
    else:
        st.session_state[key] = effective


def init_page_ui_from_template(
    *,
    initial_preset_id: str,
    template: FilmstackTemplate | None,
    ui: FilmstackUIApplySpec,
    defaults: FilmstackUIDefaults,
) -> None:
    """Apply template sim to session before first widget render."""
    init_key = f"{ui.wl_from_key}__ui_inited_{initial_preset_id}"
    if st.session_state.get(init_key):
        return
    if initial_preset_id == CUSTOM_PRESET_ID:
        sim = FilmstackSimParams()
    else:
        sim = template.sim if template is not None else FilmstackSimParams()
    _apply_ui_params(ui, sim, defaults)
    st.session_state[init_key] = True


def _apply_ui_params(
    ui: FilmstackUIApplySpec,
    sim: FilmstackSimParams,
    defaults: FilmstackUIDefaults,
) -> None:
    _apply_sim_param(ui.wl_from_key, sim.wl_from_um, default=defaults.wl_from, fmt="%.4f")
    _apply_sim_param(ui.wl_to_key, sim.wl_to_um, default=defaults.wl_to, fmt="%.4f")
    _apply_sim_param(ui.ang_from_key, sim.ang_from_deg, default=defaults.ang_from, fmt="%.2f")
    _apply_sim_param(ui.ang_to_key, sim.ang_to_deg, default=defaults.ang_to, fmt="%.2f")
    _apply_sim_param(ui.target_wl_key, sim.target_wl_um, default=defaults.target_wl, fmt="%.4f")
    _apply_sim_param(ui.target_ang_key, sim.target_ang_deg, default=defaults.target_ang, fmt="%.2f")
    if ui.fixed_angle_key is not None:
        angle = sim.target_ang_deg
        if angle is None:
            angle = sim.ang_from_deg
        _apply_sim_param(ui.fixed_angle_key, angle, default=defaults.fixed_angle, fmt="%.2f")
    _apply_sim_param(ui.polarization_key, sim.polarization, default=defaults.polarization)


def apply_preset_template(
    template: FilmstackTemplate | None,
    preset_id: str,
    catalog: PresetCatalog,
    *,
    formula_key: str,
    preset_key: str,
    ui: FilmstackUIApplySpec,
    defaults: FilmstackUIDefaults,
) -> None:
    if preset_id == CUSTOM_PRESET_ID:
        _apply_ui_params(ui, FilmstackSimParams(), defaults)
        st.session_state[preset_key] = CUSTOM_PRESET_ID
        return

    st.session_state[formula_key] = (
        template.preset.formula if template is not None
        else build_formula_for_preset(preset_id, catalog)
    )
    st.session_state[preset_key] = preset_id

    sim = template.sim if template is not None else FilmstackSimParams()
    _apply_ui_params(ui, sim, defaults)


def resolve_stack_cached(
    formula: str,
    db: Dict[str, Any],
    *,
    cache_key: str,
    resolve: Callable[[str, Dict[str, Any]], tuple[Any, ...]],
) -> tuple[Any, ...]:
    """Resolve formula once per session until formula changes."""
    key = formula.strip()
    cached = st.session_state.get(cache_key)
    if cached and cached.get("key") == key:
        return tuple(cached["values"])
    values = resolve(key, db)
    st.session_state[cache_key] = {"key": key, "values": values}
    return values


def on_preset_change(
    *,
    preset_select_key: str,
    preset_key: str,
    formula_key: str,
    page_context_key: str,
    preset_ids: tuple[str, ...],
    ui: FilmstackUIApplySpec | None = None,
    defaults_factory: Callable[[Any], FilmstackUIDefaults] | None = None,
) -> None:
    ctx = st.session_state.get(page_context_key)
    if ctx is None:
        return
    idx = st.session_state.get(preset_select_key, 0)
    if not (0 <= idx < len(preset_ids)):
        return
    preset_id = preset_ids[idx]
    template = ctx.template_by_id.get(preset_id) if ui is not None else None
    defaults = None
    if ui is not None and defaults_factory is not None:
        defaults = defaults_factory(ctx)
        apply_preset_template(
            template,
            preset_id,
            ctx.preset_catalog,
            formula_key=formula_key,
            preset_key=preset_key,
            ui=ui,
            defaults=defaults,
        )


def make_preset_change_handler(
    *,
    preset_changed_key: str,
    page_context_key: str,
    preset_select_key: str,
    preset_key: str,
    formula_key: str,
    ui: FilmstackUIApplySpec,
    defaults_factory: Callable[[Any], FilmstackUIDefaults],
) -> Callable[[], None]:
    def handler() -> None:
        st.session_state[preset_changed_key] = True
        ctx = st.session_state.get(page_context_key)
        if ctx is None:
            return
        on_preset_change(
            preset_select_key=preset_select_key,
            preset_key=preset_key,
            formula_key=formula_key,
            page_context_key=page_context_key,
            preset_ids=ctx.preset_catalog.preset_ids,
            ui=ui,
            defaults_factory=defaults_factory,
        )

    return handler


def resolve_initial_formula(
    *,
    initial_preset_id: str,
    initial_formula: str,
    preset_catalog: PresetCatalog,
) -> str:
    if initial_preset_id == CUSTOM_PRESET_ID:
        return initial_formula
    return build_formula_for_preset(initial_preset_id, preset_catalog)


def init_preset_select(
    *,
    preset_key: str,
    preset_select_key: str,
    preset_ids: tuple[str, ...],
    default_preset_id: str,
) -> None:
    if preset_key not in st.session_state or st.session_state[preset_key] not in preset_ids:
        st.session_state[preset_key] = default_preset_id
    if preset_select_key not in st.session_state:
        preset_id = st.session_state[preset_key]
        st.session_state[preset_select_key] = (
            preset_ids.index(preset_id) if preset_id in preset_ids else 0
        )


def init_formula_default(
    *,
    formula_key: str,
    default_formula: str,
) -> None:
    if formula_key not in st.session_state:
        st.session_state[formula_key] = default_formula
