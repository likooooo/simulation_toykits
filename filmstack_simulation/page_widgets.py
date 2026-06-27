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
from filmstack_simulation.presets import CUSTOM_PRESET_ID, PresetCatalog, build_formula_for_preset, get_wl_mid_um

FILMSTACK_TOKENS_CSS_KEY = "_filmstack_tokens_css"

POLARIZATION_IDS = ("TE", "TM", "UNPOLARIZED")
POLARIZATION_LABELS = {"TE": "TE", "TM": "TM", "UNPOLARIZED": "Unpolarized"}
DEFAULT_POLARIZATION = "UNPOLARIZED"

# Param stack row count (sim page: 5 rows; opt page adds spacer to match).
PARAM_STACK_ROW_COUNT = 5
# Left formula text area height (px); fixed to align with param stack visually.
FORMULA_STACK_HEIGHT_PX = 300
_PARAM_ROW_COLS = [1.15, 1, 1]


def _format_float(value: float, fmt: str) -> str:
    """Printf-style formatting matching ``st.number_input(format=...)``."""
    return fmt % float(value)


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
    align: Literal["left", "right"] = "left",
    key: str | None = None,
) -> None:
    panel_section_head(
        label,
        help_text=help_text,
        help_url=help_url,
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
    title, c_from, c_to = st.columns(_PARAM_ROW_COLS, gap="small")
    with title:
        st.markdown(
            f'<div class="{css_prefix}-panel-head">{label}</div>',
            unsafe_allow_html=True,
        )
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
    title, c_wl, c_ang = st.columns(_PARAM_ROW_COLS, gap="small")
    with title:
        st.markdown(
            f'<div class="{css_prefix}-panel-head">{label}</div>',
            unsafe_allow_html=True,
        )
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


def single_input_row(
    label: str,
    *,
    css_prefix: str,
    key: str,
    default: float,
    fmt: str,
) -> float:
    title, c_val, c_extra = st.columns(_PARAM_ROW_COLS, gap="small")
    with title:
        st.markdown(
            f'<div class="{css_prefix}-panel-head">{label}</div>',
            unsafe_allow_html=True,
        )
    with c_val:
        value = _float_text_input(
            label,
            key=key,
            default=default,
            fmt=fmt,
        )
    with c_extra:
        st.empty()
    return value


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
    title, c_from, c_to = st.columns(_PARAM_ROW_COLS, gap="small")
    with title:
        st.empty()
    with c_from:
        st.empty()
    with c_to:
        st.empty()


def preset_polarization_row(
    *,
    preset_options: Sequence[Any],
    preset_format_func: Callable[[Any], str],
    preset_key: str,
    preset_on_change: Callable[[], None] | None,
    polarization_key: str,
    polarization_on_change: Callable[[], None] | None = None,
    css_prefix: str,
) -> str:
    st.markdown(
        f'<div class="{css_prefix}-preset-pol-row-marker"></div>',
        unsafe_allow_html=True,
    )
    c_title, c_preset, c_pol = st.columns(_PARAM_ROW_COLS, gap="small")
    with c_title:
        st.empty()
    with c_preset:
        preset_kwargs: dict[str, Any] = {
            "label": "预设膜系",
            "options": preset_options,
            "format_func": preset_format_func,
            "key": preset_key,
            "label_visibility": "collapsed",
        }
        if preset_on_change is not None:
            preset_kwargs["on_change"] = preset_on_change
        st.selectbox(**preset_kwargs)
    with c_pol:
        return polarization_select(
            key=polarization_key,
            on_change=polarization_on_change,
        )


def set_preset_formula(
    preset_id: str,
    materials_db: Dict[str, Any],
    catalog: PresetCatalog,
    *,
    formula_key: str,
    preset_key: str,
    sim_wl_from: float | None = None,
    sim_wl_to: float | None = None,
) -> None:
    if preset_id == CUSTOM_PRESET_ID:
        st.session_state[preset_key] = CUSTOM_PRESET_ID
        return
    wl_mid = get_wl_mid_um(sim_wl_from, sim_wl_to)
    st.session_state[formula_key] = build_formula_for_preset(
        preset_id, catalog, materials_db, wl_mid
    )
    st.session_state[preset_key] = preset_id


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
) -> None:
    ctx = st.session_state.get(page_context_key)
    if ctx is None:
        return
    idx = st.session_state.get(preset_select_key, 0)
    if not (0 <= idx < len(preset_ids)):
        return
    preset_id = preset_ids[idx]
    st.session_state[preset_key] = preset_id
    if preset_id == CUSTOM_PRESET_ID:
        return
    set_preset_formula(
        preset_id,
        ctx.get_materials_db(),
        ctx.preset_catalog,
        formula_key=formula_key,
        preset_key=preset_key,
        sim_wl_from=ctx.sim_wl_from,
        sim_wl_to=ctx.sim_wl_to,
    )


def resolve_initial_formula(
    *,
    initial_preset_id: str,
    initial_formula: str,
    preset_catalog: PresetCatalog,
    materials_db: Optional[Dict[str, Any]],
    sim_wl_from: float | None = None,
    sim_wl_to: float | None = None,
) -> str:
    if initial_preset_id == CUSTOM_PRESET_ID:
        return initial_formula
    if materials_db:
        wl_mid = get_wl_mid_um(sim_wl_from, sim_wl_to)
        return build_formula_for_preset(
            initial_preset_id, preset_catalog, materials_db, wl_mid
        )
    return ""


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
    materials_db: Optional[Dict[str, Any]],
    default_formula: str,
) -> None:
    if formula_key not in st.session_state:
        st.session_state[formula_key] = default_formula if materials_db else ""
