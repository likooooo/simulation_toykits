"""Regression tests for filmstack_simulation page helpers."""

from __future__ import annotations


from test.conftest import STANDARD_AR_FORMULA


def test_resolve_stack_cached_returns_three_values(monkeypatch, materials_db) -> None:
    """1D slice path unpacks materials, thicknesses_um, and layers."""
    import streamlit as st

    from filmstack_simulation.page import STACK_RESOLVED_KEY, _resolve_stack_cached

    state: dict[str, object] = {}
    monkeypatch.setattr(st, "session_state", state)

    materials, thicknesses_um, layers = _resolve_stack_cached(STANDARD_AR_FORMULA, materials_db)
    assert len(materials) == len(thicknesses_um)
    assert len(layers) == len(materials)
    assert state[STACK_RESOLVED_KEY]["key"] == STANDARD_AR_FORMULA.strip()

    # 1D draw path only needs materials and thicknesses_um
    materials2, thicknesses_um2, _ = _resolve_stack_cached(STANDARD_AR_FORMULA, materials_db)
    assert materials2 is materials
    assert thicknesses_um2 is thicknesses_um
