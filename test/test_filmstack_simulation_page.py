"""Regression tests for filmstack_simulation page helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from filmstack_simulation.filmstack_optimization.local_search.opt_config import (
    get_freehand_initial_formula,
)
from filmstack_simulation.page import (
    FORMULA_KEY,
    PRESET_KEY,
    ensure_session_defaults,
)
from filmstack_simulation.page_shell import PageContext
from filmstack_simulation.presets import CUSTOM_PRESET_ID
from test.conftest import STANDARD_AR_FORMULA
from toykits_config import (
    FILMSTACK_PRESET_CATALOG,
    RECOMMENDED_SIM_WL_FROM_UM,
    RECOMMENDED_SIM_WL_TO_UM,
    resolve_filmstack_initial_defaults,
)


def test_page_context_requires_tokens_path() -> None:
    tokens = Path("/tmp/design_tokens.css")
    ctx = PageContext(
        get_materials_db=dict,
        preset_catalog=FILMSTACK_PRESET_CATALOG,
        tokens_path=tokens,
    )
    assert ctx.tokens_path == tokens


def test_resolve_filmstack_initial_defaults_includes_recommended_wl_range() -> None:
    initial = resolve_filmstack_initial_defaults(FILMSTACK_PRESET_CATALOG.valid_preset_ids)
    assert initial.wl_from_um == RECOMMENDED_SIM_WL_FROM_UM
    assert initial.wl_to_um == RECOMMENDED_SIM_WL_TO_UM


def test_resolve_filmstack_initial_defaults_uses_template_sim_wl(monkeypatch) -> None:
    from common import get_filmstack_template_by_id

    template_map = get_filmstack_template_by_id()
    monkeypatch.setattr(
        "toykits_config.get_freehand_initial_preset_id",
        lambda _valid: "fs_wide_bp",
    )
    initial = resolve_filmstack_initial_defaults(
        FILMSTACK_PRESET_CATALOG.valid_preset_ids,
        template_by_id=template_map,
    )
    tpl = template_map["fs_wide_bp"]
    assert initial.preset_id == "fs_wide_bp"
    assert initial.wl_from_um == tpl.sim.wl_from_um
    assert initial.wl_to_um == tpl.sim.wl_to_um


def test_ensure_session_defaults_uses_app_initial_stack(mock_streamlit_session) -> None:
    state = mock_streamlit_session

    initial_formula = get_freehand_initial_formula()
    ensure_session_defaults(
        preset_catalog=FILMSTACK_PRESET_CATALOG,
        initial_preset_id=CUSTOM_PRESET_ID,
        initial_formula=initial_formula,
    )

    assert state[PRESET_KEY] == CUSTOM_PRESET_ID
    assert FORMULA_KEY in state
    assert str(state[FORMULA_KEY]).strip()


def test_initial_preset_applies_template_sim(mock_streamlit_session) -> None:
    from common import get_filmstack_template_by_id
    from filmstack_simulation.page_shell import ensure_filmstack_session_defaults, FilmstackSessionKeys
    from filmstack_simulation.page_widgets import SIM_UI_APPLY, sim_ui_defaults

    state = mock_streamlit_session
    template_map = get_filmstack_template_by_id()
    tpl = template_map["fs_al_mirror"]
    keys = FilmstackSessionKeys(
        formula_key=FORMULA_KEY,
        preset_key=PRESET_KEY,
        preset_select_key="fs_sim_preset_select",
        polarization_key="fs_sim_polarization",
        page_context_key="_test_ctx",
    )
    ensure_filmstack_session_defaults(
        keys=keys,
        preset_catalog=FILMSTACK_PRESET_CATALOG,
        initial_preset_id="fs_al_mirror",
        initial_formula="",
        template_by_id=template_map,
        ui=SIM_UI_APPLY,
        ui_defaults=sim_ui_defaults(
            wl_from=RECOMMENDED_SIM_WL_FROM_UM,
            wl_to=RECOMMENDED_SIM_WL_TO_UM,
        ),
    )
    assert state["fs_sim_polarization"] == tpl.sim.polarization
    assert float(state["fs_sim_target_ang"]) == pytest.approx(float(tpl.sim.target_ang_deg))


def test_resolve_stack_cached_returns_three_values(mock_streamlit_session, materials_db) -> None:
    """1D slice path unpacks materials, thicknesses_um, and layers."""
    from filmstack_simulation.page import STACK_RESOLVED_KEY
    from filmstack_simulation.page_widgets import resolve_stack_cached
    from filmstack_simulation.simulation import resolve_stack_with_layers

    state = mock_streamlit_session

    materials, thicknesses_um, layers = resolve_stack_cached(
        STANDARD_AR_FORMULA,
        materials_db,
        cache_key=STACK_RESOLVED_KEY,
        resolve=resolve_stack_with_layers,
    )
    assert len(materials) == len(thicknesses_um)
    assert len(layers) == len(materials)
    assert state[STACK_RESOLVED_KEY]["key"] == STANDARD_AR_FORMULA.strip()

    materials2, thicknesses_um2, _ = resolve_stack_cached(
        STANDARD_AR_FORMULA,
        materials_db,
        cache_key=STACK_RESOLVED_KEY,
        resolve=resolve_stack_with_layers,
    )
    assert materials2 is materials
    assert thicknesses_um2 is thicknesses_um
