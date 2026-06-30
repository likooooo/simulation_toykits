"""Compare spec vs JSON sim alignment and preset refresh reproduction."""

from __future__ import annotations

import numpy as np
import pytest

from filmstack_simulation.page_widgets import (
    DEFAULT_ANG_FROM,
    DEFAULT_ANG_TO,
    DEFAULT_POLARIZATION,
    DEFAULT_TARGET_ANG,
    DEFAULT_TARGET_WL,
    SIM_UI_APPLY,
    apply_preset_template,
    sim_ui_defaults,
)
from filmstack_simulation.presets import CUSTOM_PRESET_ID
from filmstack_simulation.template_types import FilmstackSimParams
from template_config import (
    INCOHERENT_TEMPLATE_IDS,
    load_all_templates,
    load_templates_json,
    polarization_from_quantity,
    template_by_id,
    validate_fs_sim_for_compare,
)
from toykits_config import FILMSTACK_PRESET_CATALOG

from scripts.build_freesnell_compare_ui import FREESNEL_COMPARE_SPECS


@pytest.fixture(scope="module")
def compare_specs() -> dict[str, dict]:
    return FREESNEL_COMPARE_SPECS


@pytest.fixture(scope="module")
def fs_templates_json() -> dict[str, dict]:
    return {
        str(t["id"]): t
        for t in load_templates_json()["templates"]
        if str(t["id"]).startswith("fs_")
    }


@pytest.fixture
def preset_apply_defaults():
    return sim_ui_defaults(wl_from=0.38, wl_to=0.78, formula="custom formula")


def test_fs_sim_has_required_fields(compare_specs, fs_templates_json) -> None:
    assert set(compare_specs) == set(fs_templates_json)
    for tid, spec in compare_specs.items():
        validate_fs_sim_for_compare(fs_templates_json[tid], x_axis=str(spec["x_axis"]))


def test_sim_polarization_matches_spec_quantity(compare_specs, fs_templates_json) -> None:
    for tid, spec in compare_specs.items():
        sim = fs_templates_json[tid]["sim"]
        assert sim["polarization"] == polarization_from_quantity(spec["quantity"]), tid


def test_compare_specs_have_no_duplicate_angle_fields(compare_specs) -> None:
    for tid, spec in compare_specs.items():
        assert "angle_deg" not in spec, tid
        assert "fixed_wl_um" not in spec, tid


def test_compare_specs_have_no_smooth_fields(compare_specs) -> None:
    for tid, spec in compare_specs.items():
        assert "smooth_sigma_um" not in spec, tid
        assert "smooth_sigma_wavenumber_cm" not in spec, tid


def test_preset_apply_yields_sim(
    mock_streamlit_session,
    preset_apply_defaults,
) -> None:
    state = mock_streamlit_session
    templates = template_by_id(load_all_templates())
    tpl = templates["fs_wide_bp"]

    apply_preset_template(
        tpl,
        "fs_wide_bp",
        FILMSTACK_PRESET_CATALOG,
        formula_key="fs_sim_formula",
        preset_key="fs_sim_preset",
        ui=SIM_UI_APPLY,
        defaults=preset_apply_defaults,
    )

    sim = tpl.sim
    assert state["fs_sim_formula"] == tpl.preset.formula
    assert state["fs_sim_wl_from"] == f"{sim.wl_from_um:.4f}"
    assert state["fs_sim_wl_to"] == f"{sim.wl_to_um:.4f}"
    assert state["fs_sim_target_ang"] == f"{sim.target_ang_deg:.2f}"
    assert state["fs_sim_polarization"] == sim.polarization


def test_preset_apply_custom_resets_to_defaults(mock_streamlit_session) -> None:
    state = mock_streamlit_session
    state["fs_sim_formula"] = "user edited formula"
    state["fs_sim_wl_from"] = "9.9999"
    defaults = sim_ui_defaults(wl_from=0.38, wl_to=0.78, formula="initial stack")

    apply_preset_template(
        None,
        CUSTOM_PRESET_ID,
        FILMSTACK_PRESET_CATALOG,
        formula_key="fs_sim_formula",
        preset_key="fs_sim_preset",
        ui=SIM_UI_APPLY,
        defaults=defaults,
    )

    assert state["fs_sim_formula"] == "user edited formula"
    assert state["fs_sim_wl_from"] == "0.3800"
    assert state["fs_sim_wl_to"] == "0.7800"
    assert state["fs_sim_ang_from"] == f"{DEFAULT_ANG_FROM:.2f}"
    assert state["fs_sim_ang_to"] == f"{DEFAULT_ANG_TO:.2f}"
    assert state["fs_sim_target_wl"] == f"{DEFAULT_TARGET_WL:.4f}"
    assert state["fs_sim_target_ang"] == f"{DEFAULT_TARGET_ANG:.2f}"
    assert state["fs_sim_polarization"] == DEFAULT_POLARIZATION
    assert state["fs_sim_preset"] == CUSTOM_PRESET_ID


def test_preset_apply_empty_sim_uses_defaults(mock_streamlit_session) -> None:
    state = mock_streamlit_session
    from filmstack_simulation.template_types import FilmstackTemplate
    from filmstack_simulation.presets import FilmstackPreset

    tpl = FilmstackTemplate(
        preset=FilmstackPreset(id="bragg_mirror", label="test", formula="air 0 glass 0"),
        sim=FilmstackSimParams(),
    )
    defaults = sim_ui_defaults(wl_from=0.38, wl_to=0.78, formula="x")

    apply_preset_template(
        tpl,
        "bragg_mirror",
        FILMSTACK_PRESET_CATALOG,
        formula_key="fs_sim_formula",
        preset_key="fs_sim_preset",
        ui=SIM_UI_APPLY,
        defaults=defaults,
    )

    assert state["fs_sim_wl_from"] == "0.3800"
    assert state["fs_sim_polarization"] == DEFAULT_POLARIZATION


@pytest.mark.parametrize("template_id", ["fs_al_mirror", "fs_wide_bp"])
def test_ui_sweep_matches_run_toykits(
    template_id: str,
    simulation,
    compare_specs,
    fs_templates_json,
) -> None:
    if template_id in INCOHERENT_TEMPLATE_IDS:
        pytest.skip("incoherent template")
    spec = compare_specs[template_id]
    if spec.get("granular_ir"):
        pytest.skip("granular IR expected mismatch")
    if spec["quantity"] == "neg_log_T":
        pytest.skip("non RT quantity")

    json_tpl = fs_templates_json[template_id]
    sim = json_tpl["sim"]

    from scripts.build_freesnell_compare_ui import run_toykits, x_display_to_wl_um

    x_axis = spec["x_axis"]
    if x_axis == "angle_deg":
        pytest.skip("angle sweep covered by sim field tests")

    wl_from = float(sim["wl_from_um"])
    wl_to = float(sim["wl_to_um"])
    x_display = np.linspace(wl_from, wl_to, 32)

    toykits_y = run_toykits(template_id, spec, json_tpl, x_display, x_axis)
    wl_um = x_display_to_wl_um(x_display, x_axis)
    toykits_y_wl = run_toykits(template_id, spec, json_tpl, wl_um, "wavelength_um")

    rmse = float(np.sqrt(np.mean((toykits_y_wl - toykits_y) ** 2)))
    assert rmse < 1e-9, f"{template_id} rmse={rmse}"
