"""Shared Streamlit page bootstrap for filmstack simulation and optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import streamlit as st

from filmstack_simulation.page_widgets import (
    DEFAULT_POLARIZATION,
    FILMSTACK_TOKENS_CSS_KEY,
    init_formula_default,
    init_preset_select,
    resolve_initial_formula,
)
from filmstack_simulation.presets import PresetCatalog

GetMaterialsDb = Callable[[], Dict[str, Any]]


@dataclass(frozen=True)
class PageContext:
    get_materials_db: GetMaterialsDb
    preset_catalog: PresetCatalog
    sim_wl_from: float | None = None
    sim_wl_to: float | None = None
    recommended_wl_from: float = 0.38
    recommended_wl_to: float = 0.78
    initial_preset_id: str = ""
    initial_formula: str = ""
    tokens_path: Path = field(kw_only=True)


@dataclass(frozen=True)
class FilmstackSessionKeys:
    formula_key: str
    preset_key: str
    preset_select_key: str
    polarization_key: str
    page_context_key: str


def ensure_filmstack_session_defaults(
    materials_db: Optional[Dict[str, Any]],
    *,
    keys: FilmstackSessionKeys,
    preset_catalog: PresetCatalog,
    initial_preset_id: str,
    initial_formula: str,
    sim_wl_from: float | None = None,
    sim_wl_to: float | None = None,
) -> None:
    init_preset_select(
        preset_key=keys.preset_key,
        preset_select_key=keys.preset_select_key,
        preset_ids=preset_catalog.preset_ids,
        default_preset_id=initial_preset_id,
    )
    if keys.formula_key not in st.session_state:
        default_formula = resolve_initial_formula(
            initial_preset_id=initial_preset_id,
            initial_formula=initial_formula,
            preset_catalog=preset_catalog,
            materials_db=materials_db,
            sim_wl_from=sim_wl_from,
            sim_wl_to=sim_wl_to,
        )
        init_formula_default(
            formula_key=keys.formula_key,
            materials_db=materials_db,
            default_formula=default_formula,
        )
    if keys.polarization_key not in st.session_state:
        st.session_state[keys.polarization_key] = DEFAULT_POLARIZATION


def bootstrap_filmstack_page(
    *,
    page_title: str,
    inject_styles: Callable[[str], None],
    context: PageContext,
    keys: FilmstackSessionKeys,
    materials_db: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], PresetCatalog, list[str], dict[str, str], float, float]:
    st.set_page_config(page_title=page_title, layout="wide")
    tokens_css = context.tokens_path.read_text(encoding="utf-8")
    st.session_state[FILMSTACK_TOKENS_CSS_KEY] = tokens_css
    inject_styles(tokens_css)
    st.session_state[keys.page_context_key] = context
    db = materials_db if materials_db is not None else context.get_materials_db()
    ensure_filmstack_session_defaults(
        db if db else None,
        keys=keys,
        preset_catalog=context.preset_catalog,
        initial_preset_id=context.initial_preset_id,
        initial_formula=context.initial_formula,
        sim_wl_from=context.sim_wl_from,
        sim_wl_to=context.sim_wl_to,
    )
    catalog = context.preset_catalog
    return (
        db,
        catalog,
        catalog.preset_ids,
        catalog.preset_labels,
        context.recommended_wl_from,
        context.recommended_wl_to,
    )
