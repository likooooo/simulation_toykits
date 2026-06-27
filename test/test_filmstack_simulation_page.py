"""Regression tests for filmstack_simulation page helpers."""

from __future__ import annotations

from pathlib import Path

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


def test_ensure_session_defaults_uses_app_initial_stack(mock_streamlit_session, materials_db) -> None:
    state = mock_streamlit_session

    initial_formula = get_freehand_initial_formula()
    ensure_session_defaults(
        materials_db,
        preset_catalog=FILMSTACK_PRESET_CATALOG,
        initial_preset_id=CUSTOM_PRESET_ID,
        initial_formula=initial_formula,
    )

    assert state[PRESET_KEY] == CUSTOM_PRESET_ID
    assert FORMULA_KEY in state
    assert str(state[FORMULA_KEY]).strip()


def test_resolve_stack_cached_returns_three_values(mock_streamlit_session, materials_db) -> None:
    """1D slice path unpacks materials, thicknesses_um, and layers."""
    from filmstack_simulation.page import STACK_RESOLVED_KEY, _resolve_stack_cached

    state = mock_streamlit_session

    materials, thicknesses_um, layers = _resolve_stack_cached(STANDARD_AR_FORMULA, materials_db)
    assert len(materials) == len(thicknesses_um)
    assert len(layers) == len(materials)
    assert state[STACK_RESOLVED_KEY]["key"] == STANDARD_AR_FORMULA.strip()

    # 1D draw path only needs materials and thicknesses_um
    materials2, thicknesses_um2, _ = _resolve_stack_cached(STANDARD_AR_FORMULA, materials_db)
    assert materials2 is materials
    assert thicknesses_um2 is thicknesses_um
