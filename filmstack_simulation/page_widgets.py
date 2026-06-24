"""Shared Streamlit widgets for filmstack simulation and optimization pages."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import streamlit as st

from filmstack_simulation.presets import CUSTOM_PRESET_ID, build_formula_for_preset, get_wl_mid_um

POLARIZATION_IDS = ("TE", "TM", "UNPOLARIZED")
POLARIZATION_LABELS = {"TE": "TE", "TM": "TM", "UNPOLARIZED": "Unpolarized"}
DEFAULT_POLARIZATION = "UNPOLARIZED"

_RANGE_COLS = [1.15, 1, 1]
_RANGE_WITH_POL_COLS = [1.15, 1, 1, 0.85]
_SINGLE_INPUT_COLS = [1, 1.8]


def panel_head(label: str, *, css_prefix: str, help_url: str | None = None) -> None:
    link = (
        f' — 📖 <a href="{help_url}" target="_blank" rel="noopener">使用说明</a>'
        if help_url
        else ""
    )
    st.markdown(
        f'<div class="{css_prefix}-panel-head">{label}{link}</div>',
        unsafe_allow_html=True,
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
    title, c_from, c_to = st.columns(_RANGE_COLS, gap="small")
    with title:
        st.markdown(
            f'<div class="{css_prefix}-panel-head">{label}</div>',
            unsafe_allow_html=True,
        )
    with c_from:
        val_from = st.number_input(
            "from",
            value=default_from,
            format=fmt,
            key=key_from,
            label_visibility="collapsed",
        )
    with c_to:
        val_to = st.number_input(
            "to",
            value=default_to,
            format=fmt,
            key=key_to,
            label_visibility="collapsed",
        )
    return val_from, val_to


def single_input(
    label: str,
    *,
    css_prefix: str,
    key: str,
    default: float,
    fmt: str,
) -> float:
    title, c_val = st.columns(_SINGLE_INPUT_COLS, gap="small")
    with title:
        st.markdown(
            f'<div class="{css_prefix}-panel-head">{label}</div>',
            unsafe_allow_html=True,
        )
    with c_val:
        return st.number_input(
            label,
            value=default,
            format=fmt,
            key=key,
            label_visibility="collapsed",
        )


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


def range_inputs_with_polarization(
    label: str,
    *,
    css_prefix: str,
    key_from: str,
    key_to: str,
    default_from: float,
    default_to: float,
    fmt: str,
    polarization_key: str,
    on_polarization_change: Callable[[], None] | None = None,
) -> tuple[float, float, str]:
    title, c_from, c_to, c_pol = st.columns(_RANGE_WITH_POL_COLS, gap="small")
    with title:
        st.markdown(
            f'<div class="{css_prefix}-panel-head">{label}</div>',
            unsafe_allow_html=True,
        )
    with c_from:
        val_from = st.number_input(
            "from",
            value=default_from,
            format=fmt,
            key=key_from,
            label_visibility="collapsed",
        )
    with c_to:
        val_to = st.number_input(
            "to",
            value=default_to,
            format=fmt,
            key=key_to,
            label_visibility="collapsed",
        )
    with c_pol:
        polarization = polarization_select(
            key=polarization_key,
            on_change=on_polarization_change,
        )
    return val_from, val_to, polarization


def set_preset_formula(
    preset_id: str,
    materials_db: Dict[str, Any],
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
    st.session_state[formula_key] = build_formula_for_preset(preset_id, materials_db, wl_mid)
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
        formula_key=formula_key,
        preset_key=preset_key,
        sim_wl_from=ctx.sim_wl_from,
        sim_wl_to=ctx.sim_wl_to,
    )


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
