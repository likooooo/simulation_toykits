"""Tests for filmstack default materials and presets."""

from __future__ import annotations

import re

from filmstack_simulation.materials import (
    RECOMMENDED_SIM_WL_FROM_UM,
    RECOMMENDED_SIM_WL_TO_UM,
)
from filmstack_simulation.presets import (
    _BRAGG_PERIOD_UM,
    PRESETS,
    build_formula_for_preset,
    get_wl_mid_um,
)
from simulation_database.database_ui import (
    intersect_wl_ranges,
    material_wl_range_um,
)


def test_default_materials_db_nonempty(materials_db) -> None:
    assert len(materials_db) == 13
    catalogs = set(materials_db)
    assert "air" in catalogs
    assert "SiO2" in catalogs
    assert "Ta2O5" in catalogs
    assert "Si" in catalogs
    assert "ito" in catalogs
    assert "MgF2" in catalogs
    assert "TiO2" in catalogs
    assert "N-BK7" in catalogs
    assert catalogs.isdisjoint({"1.5", "3.0", "1.0"})


def test_material_wavelength_intersection_covers_recommended_range(materials_db) -> None:
    ranges = [material_wl_range_um(mat) for mat in materials_db.values()]
    intersection = intersect_wl_ranges(ranges)
    assert intersection is not None
    lo, hi = intersection
    assert lo <= RECOMMENDED_SIM_WL_FROM_UM
    assert hi >= RECOMMENDED_SIM_WL_TO_UM


def test_ar_qw_formula_uses_db_materials(materials_db) -> None:
    wl_mid = get_wl_mid_um(RECOMMENDED_SIM_WL_FROM_UM, RECOMMENDED_SIM_WL_TO_UM)
    formula = build_formula_for_preset("ar_qw_si", materials_db, wl_mid)
    assert "1.46" not in formula
    assert "2.1" not in formula
    assert "SiO2" in formula
    assert "Ta2O5" in formula
    assert "^2" in formula


def test_bragg_period_conserved(materials_db) -> None:
    wl_mid = 0.55
    formula = build_formula_for_preset("bragg_mirror", materials_db, wl_mid)
    assert "^30" in formula
    assert "(H " in formula
    match = re.search(r"\(H ([0-9.]+) 1\.5 0\.0 air ([0-9.]+)\)\^30", formula)
    assert match is not None
    t_h = float(match.group(1))
    t_l = float(match.group(2))
    assert abs(t_h + t_l - _BRAGG_PERIOD_UM) < 1e-6


_PRESET_LAYER_COUNTS = {
    "ar_qw_si": 6,
    "bragg_mirror": 62,
    "optical_filter": 14,
    "fabry_perot": 5,
    "oled_ito_al": 8,
    "spr_bk7_cr_au": 4,
    "paper_sio2_si": 3,
}

_NUMERIC_CATALOG_TOKENS = frozenset({"1.5", "3.0", "1.0"})


def test_preset_formula_layer_counts(preset_parsed_layers) -> None:
    for preset in PRESETS:
        parsed = preset_parsed_layers[preset.id]
        assert len(parsed) == _PRESET_LAYER_COUNTS[preset.id], (
            f"{preset.id}: expected {_PRESET_LAYER_COUNTS[preset.id]} layers, got {len(parsed)}"
        )
        materials = {layer["Material"] for layer in parsed}
        assert materials.isdisjoint(_NUMERIC_CATALOG_TOKENS), (
            f"{preset.id}: numeric catalog tokens found in parsed materials: {materials}"
        )


def test_paper_preset_inline_nk() -> None:
    formula = build_formula_for_preset("paper_sio2_si", {}, 0.633)
    assert "1.46" in formula
    assert "3.87" in formula
    assert "0.02" in formula


def test_spr_preset_inline_nk() -> None:
    formula = build_formula_for_preset("spr_bk7_cr_au", {}, 0.633)
    assert "1.517" in formula
    assert "3.719" in formula
    assert "4.362" in formula
    assert "0.13" in formula
    assert "3.162" in formula
