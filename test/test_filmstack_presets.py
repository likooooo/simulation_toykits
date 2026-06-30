"""Tests for filmstack default materials and presets."""

from __future__ import annotations

import re

import pytest

from filmstack_simulation.presets import build_formula_for_preset
from simulation_database.database_ui import (
    intersect_wl_ranges,
    material_wl_range_um,
)
from toykits_config import (
    FILMSTACK_PRESET_CATALOG,
    RECOMMENDED_SIM_WL_FROM_UM,
    RECOMMENDED_SIM_WL_TO_UM,
)

BRAGG_PERIOD_UM = 0.209
BRAGG_PERIOD_REPEAT = 8


def _assert_mg_layer(
    formula: str,
    *,
    q: float,
    inclusion: str | None = None,
    host_name: str | None = None,
    host_inline_n: float | None = None,
) -> dict:
    from filmstack_visualizer import parse_filmstack_formula_v1

    parsed = parse_filmstack_formula_v1(formula)
    mg = next(layer for layer in parsed if layer["Material"] == "maxwell_garnett")
    assert mg["mg_q"] == pytest.approx(q)
    if inclusion is not None:
        assert mg["mg_inclusion"] == inclusion
    if host_name is not None:
        assert mg["mg_host_name"] == host_name
    if host_inline_n is not None:
        assert mg["mg_host_inline_n"] == pytest.approx(host_inline_n)
    return mg


def test_default_materials_db_nonempty(materials_db) -> None:
    assert len(materials_db) >= 13
    unique_names = set(materials_db)
    assert "air_Ciddor" in unique_names
    assert "SiO2_Arosa" in unique_names
    assert "Ta2O5_Cheikh-amorphous-3.28-8-450" in unique_names
    assert "Si_Aspnes" in unique_names
    assert "ito" in unique_names
    assert "MgF2_Dodge-o" in unique_names
    assert "TiO2_Jolivet-anatase" in unique_names
    assert "N-BK7" in unique_names
    assert unique_names.isdisjoint({"1.5", "3.0", "1.0"})


def test_material_wavelength_intersection_covers_recommended_range(materials_db) -> None:
    legacy = {
        "air_Ciddor",
        "SiO2_Arosa",
        "Ta2O5_Cheikh-amorphous-3.28-8-450",
        "Si_Aspnes",
        "MgF2_Dodge-o",
        "TiO2_Jolivet-anatase",
        "N-BK7",
    }
    ranges = [
        material_wl_range_um(mat)
        for name, mat in materials_db.items()
        if name in legacy
    ]
    intersection = intersect_wl_ranges(ranges)
    assert intersection is not None
    lo, hi = intersection
    assert lo <= RECOMMENDED_SIM_WL_FROM_UM
    assert hi >= RECOMMENDED_SIM_WL_TO_UM


def test_ar_qw_formula_uses_db_materials() -> None:
    formula = build_formula_for_preset("ar_qw_si", FILMSTACK_PRESET_CATALOG)
    assert "1.46" not in formula
    assert "2.1" not in formula
    assert "SiO2_Arosa" in formula
    assert "Ta2O5_Cheikh-amorphous-3.28-8-450" in formula
    assert "^2" in formula


def test_bragg_period_conserved() -> None:
    formula = build_formula_for_preset("bragg_mirror", FILMSTACK_PRESET_CATALOG)
    assert f"^{BRAGG_PERIOD_REPEAT}" in formula
    assert "(H " in formula
    match = re.search(
        rf"\(H ([0-9.]+) 1\.5 0\.0 air_Ciddor ([0-9.]+)\)\^{BRAGG_PERIOD_REPEAT}",
        formula,
    )
    assert match is not None
    t_h = float(match.group(1))
    t_l = float(match.group(2))
    assert abs(t_h + t_l - BRAGG_PERIOD_UM) < 1e-6


_LEGACY_PRESET_LAYER_COUNTS = {
    "ar_qw_si": 6,
    "bragg_mirror": 18,
    "optical_filter": 14,
    "fabry_perot": 5,
    "oled_ito_al": 8,
    "spr_bk7_cr_au": 4,
    "paper_sio2_si": 3,
}

_NUMERIC_INLINE_TOKENS = frozenset({"1.5", "3.0", "1.0"})


def test_preset_formula_layer_counts(preset_parsed_layers) -> None:
    for preset_id, expected in _LEGACY_PRESET_LAYER_COUNTS.items():
        parsed = preset_parsed_layers[preset_id]
        assert len(parsed) == expected, (
            f"{preset_id}: expected {expected} layers, got {len(parsed)}"
        )
        materials = {layer["Material"] for layer in parsed}
        assert materials.isdisjoint(_NUMERIC_INLINE_TOKENS), (
            f"{preset_id}: numeric inline tokens found in parsed materials: {materials}"
        )


def test_paper_preset_inline_nk() -> None:
    formula = build_formula_for_preset("paper_sio2_si", FILMSTACK_PRESET_CATALOG)
    assert "1.46" in formula
    assert "3.87" in formula
    assert "0.02" in formula


def test_spr_preset_inline_nk() -> None:
    formula = build_formula_for_preset("spr_bk7_cr_au", FILMSTACK_PRESET_CATALOG)
    assert "1.517" in formula
    assert "3.719" in formula
    assert "4.362" in formula
    assert "0.13" in formula
    assert "3.162" in formula


def test_ruby_glass_presets_use_maxwell_garnett_syntax() -> None:
    formulas = {
        "fs_ruby_glass_2um": 2e-6,
        "fs_ruby_glass_8um": 8e-6,
        "fs_ruby_glass_14um": 14e-6,
    }
    built = []
    for preset_id, q in formulas.items():
        formula = build_formula_for_preset(preset_id, FILMSTACK_PRESET_CATALOG)
        built.append(formula)
        _assert_mg_layer(
            formula,
            q=q,
            inclusion="au",
            host_name="glass",
            host_inline_n=1.5,
        )
    assert len(set(built)) == 3


def test_granular_ir_presets_use_maxwell_garnett_syntax() -> None:
    ag_mgo_ids = ("fs_ag_mgo_3nm", "fs_ag_mgo_s", "fs_ag_mgo_ang_film")
    for preset_id in ag_mgo_ids:
        formula = build_formula_for_preset(preset_id, FILMSTACK_PRESET_CATALOG)
        _assert_mg_layer(formula, q=0.67, inclusion="ag", host_inline_n=1.0)

    p_formula = build_formula_for_preset("fs_ag_mgo_p", FILMSTACK_PRESET_CATALOG)
    assert "mgo 0.001" in p_formula
    assert " mgo 0 " not in f" {p_formula} "
    _assert_mg_layer(p_formula, q=0.67, inclusion="ag", host_inline_n=1.0)

    polymer_formula = build_formula_for_preset("fs_polymer_ag", FILMSTACK_PRESET_CATALOG)
    assert "0.166667" in polymer_formula
    assert "0.083333" in polymer_formula
    from filmstack_visualizer import parse_filmstack_formula_v1

    polymer_parsed = parse_filmstack_formula_v1(polymer_formula)
    mg_layers = [layer for layer in polymer_parsed if layer["Material"] == "maxwell_garnett"]
    assert len(mg_layers) > 1
    q_vals = {round(layer["mg_q"], 6) for layer in mg_layers}
    assert q_vals == {0.166667, 0.083333}

    nobles_transitions = {
        "fs_nobles_ag": "ag",
        "fs_nobles_au": "au",
        "fs_nobles_cu": "cu",
        "fs_nobles_al": "al",
        "fs_transitions_co": "co",
        "fs_transitions_pt": "pt",
        "fs_transitions_pd": "pd",
        "fs_transitions_ti": "ti",
    }
    for preset_id, metal in nobles_transitions.items():
        formula = build_formula_for_preset(preset_id, FILMSTACK_PRESET_CATALOG)
        assert f"[0.67 {metal} air 1.0 0.0]" in formula


def test_granular_ir_presets_resolve_stack(sim_db, simulation) -> None:
    from filmstack_visualizer import layers_from_formula
    from simulation_database.database_precompiling import get_precompiled_leaf_object
    from simulation_database.database_ui import object_unique_name

    fs_paths = (
        (("fs", "materials", "ag.yml"), "ag"),
        (("fs", "materials", "mgo.yml"), "mgo"),
        (("fs", "materials", "au.yml"), "au"),
        (("fs", "materials", "co.yml"), "co"),
    )
    materials_db: dict[str, object] = {}
    for path_keys, alias in fs_paths:
        obj = get_precompiled_leaf_object(path_keys)
        materials_db[alias] = obj
        materials_db[object_unique_name(obj)] = obj

    for preset_id in ("fs_ag_mgo_3nm", "fs_nobles_ag", "fs_transitions_co"):
        formula = build_formula_for_preset(preset_id, FILMSTACK_PRESET_CATALOG)
        mats, th = layers_from_formula(formula, materials_db, simulation_module=simulation)
        mg_mats = [
            mat
            for mat in mats
            if mat.active_model == simulation.material_model_kind.maxwell_garnett
        ]
        assert len(mg_mats) >= 1
        assert all(float(t) >= 0.0 for t in th)


def test_fs_immersed_polarizer_symmetric_bk7_bookends(simulation) -> None:
    from filmstack_visualizer import layers_from_formula

    formula = build_formula_for_preset("fs_immersed_polarizer", FILMSTACK_PRESET_CATALOG)
    assert formula.startswith("BK7 0 1.5164 0.0")
    assert formula.endswith("BK7 0 1.5164 0.0")
    assert "air 0" not in formula.split()[:3]

    mats, th = layers_from_formula(formula, {}, simulation_module=simulation)
    assert th[0] == 0.0 and th[-1] == 0.0
    assert len(th) == 19  # BK7 + 17 film layers + BK7
    assert mats[0].name == "BK7"
    assert mats[-1].name == "BK7"
    assert sum(1 for t in th if t > 0.0) == 17


def test_templates_json_loads() -> None:
    from template_config import load_all_templates, load_templates_json

    doc = load_templates_json()
    templates = load_all_templates()
    assert len(templates) == len(doc["templates"])
    for t in templates:
        assert t.preset.formula.strip()
        assert " wl" not in f" {t.preset.formula} "
        assert t.preset.formula.startswith("wl") is False


def test_database_precompiled_round_trip(sim_db) -> None:
    from simulation_database.database_precompiling import (
        get_active_index,
        get_precompiled_leaf_object,
        load_or_build_database_index,
        panel_search_catalog,
        precompiled_bundle_path,
    )
    from simulation_database.database_ui import get_tree_children, object_unique_name, search_db_paths
    from toykits_config import DEFAULT_MATERIAL_PATH_KEYS

    assert precompiled_bundle_path().is_file()
    index = load_or_build_database_index(sim_db)
    assert get_active_index() is index
    assert index.leaf_count > 0
    assert index.inverted_index
    assert len(index.leaf_objects_by_path_id) == index.leaf_count

    catalog = panel_search_catalog(sim_db)
    assert catalog["entries"]
    assert catalog["inverted"]
    assert catalog.get("fingerprint")

    assert search_db_paths(sim_db, "Si")
    assert get_tree_children(sim_db, [], {})

    path_keys = DEFAULT_MATERIAL_PATH_KEYS[0]
    obj = get_precompiled_leaf_object(path_keys)
    assert object_unique_name(obj)


def test_workspace_no_path_keys_does_not_mark_defaults_loaded(sim_db, mock_streamlit_session) -> None:
    from simulation_database.database_precompiling import load_or_build_database_index
    from simulation_database.workspace import ensure_sim_workspace_ui, ensure_workspace_initialized

    load_or_build_database_index(sim_db)
    ws, ui = ensure_workspace_initialized(sim_db)
    assert not ui.defaults_loaded
    assert len(ws.materials) == 0


def test_workspace_strict_loads_required_materials(sim_db, mock_streamlit_session) -> None:
    from simulation_database.database_precompiling import load_or_build_database_index
    from simulation_database.workspace import ensure_workspace_initialized, get_workspace_materials
    from toykits_config import (
        DEFAULT_MATERIAL_PATH_KEYS,
        DEFAULT_SPECTRUM_PATH,
        REQUIRED_DEFAULT_MATERIAL_NAMES,
    )

    load_or_build_database_index(sim_db)
    ensure_workspace_initialized(
        sim_db,
        material_path_keys=DEFAULT_MATERIAL_PATH_KEYS,
        spectrum_path_keys=[DEFAULT_SPECTRUM_PATH],
        strict=True,
        required_material_names=REQUIRED_DEFAULT_MATERIAL_NAMES,
    )
    materials = get_workspace_materials()
    assert set(materials) >= REQUIRED_DEFAULT_MATERIAL_NAMES
    assert len(materials) >= len(REQUIRED_DEFAULT_MATERIAL_NAMES)


def test_workspace_schema_reset_allows_reload(sim_db, mock_streamlit_session) -> None:
    from simulation_database.database_precompiling import load_or_build_database_index
    from simulation_database.workspace import (
        SimWorkspace,
        ensure_sim_workspace_ui,
        ensure_workspace_initialized,
        get_workspace_materials,
    )
    from toykits_config import (
        DEFAULT_MATERIAL_PATH_KEYS,
        DEFAULT_SPECTRUM_PATH,
        REQUIRED_DEFAULT_MATERIAL_NAMES,
    )

    load_or_build_database_index(sim_db)
    init_kwargs = {
        "material_path_keys": DEFAULT_MATERIAL_PATH_KEYS,
        "spectrum_path_keys": [DEFAULT_SPECTRUM_PATH],
        "strict": True,
        "required_material_names": REQUIRED_DEFAULT_MATERIAL_NAMES,
    }
    ensure_workspace_initialized(sim_db, **init_kwargs)
    assert len(get_workspace_materials()) >= len(REQUIRED_DEFAULT_MATERIAL_NAMES)

    ui = ensure_sim_workspace_ui()
    ui.defaults_loaded = False
    mock_streamlit_session["sim_workspace"] = SimWorkspace()

    ensure_workspace_initialized(sim_db, **init_kwargs)
    assert set(get_workspace_materials()) >= REQUIRED_DEFAULT_MATERIAL_NAMES
