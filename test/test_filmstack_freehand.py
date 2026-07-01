"""Tests for Freehand optimization helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from filmstack_simulation.filmstack_optimization.local_search.page import build_freehand_config
from filmstack_simulation.filmstack_optimization.local_search.page import consume_component_event  # noqa: E402
from filmstack_simulation.filmstack_optimization.local_search.freehand_state import (  # noqa: E402
    FreehandSession,
    auto_y_max_percent,
    auto_y_min_percent,
    default_view_domain,
)
from filmstack_simulation.filmstack_optimization.local_search.opt_config import (  # noqa: E402
    get_freehand_cost_scope,
    get_freehand_initial_formula,
    get_freehand_initial_preset_id,
    get_freehand_n_wl,
    load_freehand_base_config,
)
from filmstack_simulation.presets import CUSTOM_PRESET_ID  # noqa: E402
from toykits_config import FILMSTACK_PRESET_CATALOG  # noqa: E402
from filmstack_simulation.simulation import combine_polarization_rt, compute_rta_at_angle, resolve_stack  # noqa: E402
from filmstack_simulation.filmstack_optimization.shared.stack_table import formula_from_stack  # noqa: E402


def test_auto_y_max_percent() -> None:
    assert auto_y_max_percent(np.array([0.5])) == pytest.approx(50.0)
    assert auto_y_max_percent(np.array([0.9])) == pytest.approx(90.0)
    assert auto_y_max_percent(np.array([0.995])) == pytest.approx(100.0)
    assert auto_y_max_percent(np.array([0.0])) == pytest.approx(1.0)
    assert auto_y_max_percent(None, np.array([0.25, 0.4])) == pytest.approx(40.0)


def test_auto_y_min_percent() -> None:
    assert auto_y_min_percent(np.array([0.5])) == pytest.approx(50.0)
    assert auto_y_min_percent(np.array([0.0])) == pytest.approx(0.0)
    assert auto_y_min_percent(None, np.array([0.25, 0.4])) == pytest.approx(25.0)


def test_default_view_domain_uses_percent() -> None:
    current = {
        "R": np.array([0.4, 0.5]),
        "T": np.array([0.1, 0.2]),
        "A": np.array([0.0, 0.0]),
    }
    dom = default_view_domain(0.4, 0.8, current=current)
    assert dom["R"]["y"] == [pytest.approx(40.0), pytest.approx(50.0)]
    assert dom["T"]["y"] == [pytest.approx(10.0), pytest.approx(20.0)]
    assert dom["A"]["y"] == [0.0, pytest.approx(1.0)]
    assert dom["R"]["yAuto"] is True


def test_refresh_auto_y_domains_preserves_manual_y_zoom() -> None:
    session = FreehandSession()
    wl = np.linspace(0.4, 0.8, 3)
    session.reset_after_build(
        formula="air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0",
        wl_um=wl,
        angle_deg=0.0,
        current={"R": np.array([0.5, 0.5, 0.5]), "T": np.zeros(3), "A": np.zeros(3)},
        wl_from=0.4,
        wl_to=0.8,
    )
    session.view_domain["R"] = {
        "x": [0.45, 0.75],
        "y": [10.0, 40.0],
        "yAuto": False,
    }
    session.current["R"] = np.array([0.8, 0.8, 0.8])
    session.refresh_auto_y_domains()
    assert session.view_domain["R"]["y"] == [10.0, 40.0]
    assert session.view_domain["R"]["yAuto"] is False
    assert session.view_domain["T"]["y"] == [0.0, pytest.approx(1.0)]


def test_apply_optimization_result_refreshes_auto_y() -> None:
    session = FreehandSession()
    wl = np.linspace(0.4, 0.8, 3)
    session.reset_after_build(
        formula="air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0",
        wl_um=wl,
        angle_deg=0.0,
        current={"R": np.array([0.2, 0.2, 0.2]), "T": np.zeros(3), "A": np.zeros(3)},
        wl_from=0.4,
        wl_to=0.8,
    )
    session.apply_optimization_result(
        formula="air_Ciddor 0 SiO2_Arosa 0.2 Si_Aspnes 0",
        current={"R": np.array([0.5, 0.5, 0.5]), "T": np.zeros(3), "A": np.zeros(3)},
        merit_history=[1.0, 0.5],
        merit_initial=1.0,
    )
    assert session.view_domain["R"]["y"] == [pytest.approx(50.0), pytest.approx(50.0)]


def test_apply_optimization_result_preserves_manual_x_zoom() -> None:
    session = FreehandSession()
    wl = np.linspace(0.4, 0.8, 3)
    session.reset_after_build(
        formula="air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0",
        wl_um=wl,
        angle_deg=0.0,
        current={"R": np.array([0.2, 0.2, 0.2]), "T": np.zeros(3), "A": np.zeros(3)},
        wl_from=0.4,
        wl_to=0.8,
    )
    session.view_domain["R"] = {
        "x": [0.45, 0.55],
        "y": [10.0, 40.0],
        "yAuto": False,
    }
    session.apply_optimization_result(
        formula="air_Ciddor 0 SiO2_Arosa 0.2 Si_Aspnes 0",
        current={"R": np.array([0.5, 0.5, 0.5]), "T": np.zeros(3), "A": np.zeros(3)},
        merit_history=[1.0, 0.5],
        merit_initial=1.0,
        pre_optimize_current={"R": np.array([0.2, 0.2, 0.2]), "T": np.zeros(3), "A": np.zeros(3)},
    )
    assert session.view_domain["R"]["x"] == [0.45, 0.55]
    assert session.view_domain["R"]["y"] == [10.0, 40.0]
    assert session.view_domain["R"]["yAuto"] is False


def test_curve_drag_end_syncs_view_domain(curve_drag_session) -> None:
    session, _wl = curve_drag_session
    from filmstack_simulation.filmstack_optimization.local_search.page import _handle_component_event

    zoomed = {
        "R": {"x": [0.45, 0.55], "y": [0.0, 100.0], "yAuto": False},
        "T": {"x": [0.4, 0.8], "y": [0.0, 1.0], "yAuto": True},
        "A": {"x": [0.4, 0.8], "y": [0.0, 1.0], "yAuto": True},
    }
    _handle_component_event(
        session,
        {
            "type": "curveDragEnd",
            "target": {"R": np.linspace(0.3, 0.6, 5).tolist(), "T": None, "A": None},
            "touched": {"R": True, "T": False, "A": False},
            "viewDomain": zoomed,
            "triggerOptimize": False,
        },
    )
    assert session.view_domain["R"]["x"] == [0.45, 0.55]
    assert session.view_domain["R"]["yAuto"] is False


def test_view_change_syncs_view_domain(curve_drag_session) -> None:
    session, _wl = curve_drag_session
    from filmstack_simulation.filmstack_optimization.local_search.page import (
        _VIEW_UI_EVENTS,
        _handle_component_event,
        _process_component_event,
    )

    zoomed = {
        "R": {"x": [0.45, 0.55], "y": [10.0, 40.0], "yAuto": False},
        "T": {"x": [0.4, 0.8], "y": [0.0, 100.0], "yAuto": False},
        "A": {"x": [0.4, 0.8], "y": [0.0, 100.0], "yAuto": False},
    }
    event = {"type": "viewChange", "viewDomain": zoomed}
    _handle_component_event(session, event)
    assert session.view_domain["R"]["x"] == [0.45, 0.55]
    assert session.view_domain["R"]["y"] == [10.0, 40.0]
    assert session.view_domain["R"]["yAuto"] is False
    assert session.view_domain["T"]["y"] == [0.0, 100.0]
    assert "viewChange" in _VIEW_UI_EVENTS
    assert (
        _process_component_event(
            session,
            event,
            {},
            film_indices=[1],
            thicknesses_um=[0.0, 0.1, 0.0],
        )
        is False
    )
    _handle_component_event(session, event)
    assert session.view_domain["R"]["x"] == [0.45, 0.55]


def test_to_component_args_includes_merit_history() -> None:
    session = FreehandSession()
    wl = np.linspace(0.4, 0.8, 3)
    session.reset_after_build(
        formula="air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0",
        wl_um=wl,
        angle_deg=0.0,
        current={"R": np.array([0.2, 0.2, 0.2]), "T": np.zeros(3), "A": np.zeros(3)},
        wl_from=0.4,
        wl_to=0.8,
    )
    assert session.to_component_args()["meritHistory"] is None

    session.apply_optimization_result(
        formula="air_Ciddor 0 SiO2_Arosa 0.2 Si_Aspnes 0",
        current={"R": np.array([0.5, 0.5, 0.5]), "T": np.zeros(3), "A": np.zeros(3)},
        merit_history=[1.0, 0.5],
        merit_initial=1.0,
    )
    assert session.to_component_args()["meritHistory"] == [1.0, 0.5]


def test_to_component_args_includes_pre_optimize_current() -> None:
    session = FreehandSession()
    wl = np.linspace(0.4, 0.8, 3)
    session.reset_after_build(
        formula="air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0",
        wl_um=wl,
        angle_deg=0.0,
        current={"R": np.array([0.2, 0.2, 0.2]), "T": np.zeros(3), "A": np.zeros(3)},
        wl_from=0.4,
        wl_to=0.8,
    )
    assert session.to_component_args()["preOptimizeCurrent"] is None

    pre = {"R": np.array([0.2, 0.2, 0.2]), "T": np.zeros(3), "A": np.zeros(3)}
    session.apply_optimization_result(
        formula="air_Ciddor 0 SiO2_Arosa 0.2 Si_Aspnes 0",
        current={"R": np.array([0.5, 0.5, 0.5]), "T": np.zeros(3), "A": np.zeros(3)},
        merit_history=[1.0, 0.5],
        merit_initial=1.0,
        pre_optimize_current=pre,
    )
    args = session.to_component_args()["preOptimizeCurrent"]
    assert args is not None
    assert args["R"] == pytest.approx([0.2, 0.2, 0.2])


def test_reset_after_build_clears_pre_optimize_current() -> None:
    session = FreehandSession()
    wl = np.linspace(0.4, 0.8, 3)
    session.reset_after_build(
        formula="air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0",
        wl_um=wl,
        angle_deg=0.0,
        current={"R": np.array([0.2, 0.2, 0.2]), "T": np.zeros(3), "A": np.zeros(3)},
        wl_from=0.4,
        wl_to=0.8,
    )
    session.apply_optimization_result(
        formula="air_Ciddor 0 SiO2_Arosa 0.2 Si_Aspnes 0",
        current={"R": np.array([0.5, 0.5, 0.5]), "T": np.zeros(3), "A": np.zeros(3)},
        merit_history=[1.0, 0.5],
        merit_initial=1.0,
        pre_optimize_current={"R": np.array([0.2, 0.2, 0.2]), "T": np.zeros(3), "A": np.zeros(3)},
    )
    session.reset_after_build(
        formula="air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0",
        wl_um=wl,
        angle_deg=0.0,
        current={"R": np.array([0.2, 0.2, 0.2]), "T": np.zeros(3), "A": np.zeros(3)},
        wl_from=0.4,
        wl_to=0.8,
    )
    assert session.pre_optimize_current is None
    assert session.to_component_args()["preOptimizeCurrent"] is None


def test_apply_optimization_result_y_domain_includes_pre_optimize_current() -> None:
    session = FreehandSession()
    wl = np.linspace(0.4, 0.8, 3)
    session.reset_after_build(
        formula="air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0",
        wl_um=wl,
        angle_deg=0.0,
        current={"R": np.array([0.2, 0.2, 0.2]), "T": np.zeros(3), "A": np.zeros(3)},
        wl_from=0.4,
        wl_to=0.8,
    )
    session.apply_optimization_result(
        formula="air_Ciddor 0 SiO2_Arosa 0.2 Si_Aspnes 0",
        current={"R": np.array([0.5, 0.5, 0.5]), "T": np.zeros(3), "A": np.zeros(3)},
        merit_history=[1.0, 0.5],
        merit_initial=1.0,
        pre_optimize_current={"R": np.array([0.1, 0.1, 0.9]), "T": np.zeros(3), "A": np.zeros(3)},
    )
    assert session.view_domain["R"]["y"] == [pytest.approx(10.0), pytest.approx(90.0)]


@pytest.mark.parametrize(
    "event,last_ts,expect_fresh",
    [
        ({"type": "curveDragEnd", "ts": 1000}, 0.0, True),
        (
            {
                "type": "curveDragEnd",
                "ts": 1000,
                "triggerOptimize": True,
                "touched": {"R": True, "T": False, "A": False},
            },
            1000.0,
            False,
        ),
    ],
    ids=["fresh_event", "stale_replay_after_build"],
)
def test_consume_component_event_deduplicates_replay(event, last_ts, expect_fresh) -> None:
    fresh, ts = consume_component_event(event, last_ts)
    if expect_fresh:
        assert fresh is event
        assert ts == event["ts"]
        stale, ts2 = consume_component_event(event, ts)
        assert stale is None
        assert ts2 == event["ts"]
    else:
        assert fresh is None
        assert ts == event["ts"]


def test_load_freehand_base_config_reads_optimizer_from_json() -> None:
    cfg = load_freehand_base_config()
    assert cfg["n_wl"] == 100
    assert cfg["freehand_cost_scope"] == "zoom"
    assert cfg["optimizer"]["options"]["maxiter"] == 8
    assert cfg["optimizer"]["options"]["maxfun"] == 12
    assert cfg["cost_function"]["name"] == "freehand_target"
    assert Path(cfg["cost_function"]["path"]).name == "cost_freehand.py"
    assert cfg["thickness_step_um"] == pytest.approx(1e-8)
    assert cfg["initial_preset_id"] == "custom"
    assert "MgF2_Dodge-o" in cfg["initial_formula"]
    assert "N-BK7" in cfg["initial_formula"]
    assert get_freehand_n_wl() == 100
    assert get_freehand_cost_scope() == "zoom"
    assert get_freehand_initial_preset_id(FILMSTACK_PRESET_CATALOG.valid_preset_ids) == CUSTOM_PRESET_ID
    formula = get_freehand_initial_formula()
    assert formula.startswith("air_Ciddor 0 MgF2_Dodge-o")
    assert formula.endswith("N-BK7 0")


def test_initial_formula_resolves_with_workspace_materials(materials_db) -> None:
    formula = get_freehand_initial_formula()
    materials, thicknesses_um = resolve_stack(formula, materials_db)
    assert len(materials) >= 9
    assert len(thicknesses_um) == len(materials)
    film_thicknesses_nm = [round(t * 1000, 2) for t in thicknesses_um if t > 0]
    assert 151.54 in film_thicknesses_nm
    assert 93.52 in film_thicknesses_nm


def test_hydrate_widgets_from_session_restores_missing_keys(mock_streamlit_session) -> None:
    state = mock_streamlit_session
    session = FreehandSession()
    wl = np.linspace(0.4, 0.8, 5)
    session.reset_after_build(
        formula="air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0",
        wl_um=wl,
        angle_deg=25.0,
        current={"R": wl * 0.1, "T": wl * 0.5, "A": wl * 0.4},
        wl_from=0.42,
        wl_to=0.78,
    )
    from filmstack_simulation.filmstack_optimization.local_search.page import _hydrate_widgets_from_session

    _hydrate_widgets_from_session(session)
    assert state["fs_opt_formula"] == "air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0"
    assert state["fs_opt_wl_from"] == pytest.approx(0.42)
    assert state["fs_opt_wl_to"] == pytest.approx(0.78)
    assert state["fs_opt_angle"] == pytest.approx(25.0)


def test_build_freehand_config_merges_runtime_over_base() -> None:
    cfg = build_freehand_config(
        working_formula="air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0",
        wl_from=0.45,
        wl_to=0.75,
        n_wl=5,
        angle_deg=15.0,
        touched={"R": False, "T": False, "A": False},
        target={"R": None, "T": None, "A": None},
    )
    assert cfg["formula"] == "air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0"
    assert cfg["target_wl"] == [0.45, 0.75, pytest.approx(0.075)]
    assert cfg["target_angle"] == [15.0, 15.0]
    assert cfg["optimizer"]["options"]["maxiter"] == 8


def test_build_freehand_config_target_wl_grid_matches_ui_points() -> None:
    import filmstack_optimization_utils as fos  # noqa: E402

    cfg = build_freehand_config(
        working_formula="air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0",
        wl_from=0.4,
        wl_to=0.8,
        n_wl=80,
        angle_deg=20.0,
        touched={"R": True, "T": False, "A": False},
        target={"R": np.linspace(0.2, 0.5, 80), "T": None, "A": None},
    )
    wls, _angles, _, _ = fos.resolve_target_axes(cfg)
    assert len(wls) == 80
    assert cfg["target_wl"][2] == pytest.approx(0.005063291139240506)


@pytest.fixture
def curve_drag_session() -> tuple[FreehandSession, np.ndarray]:
    session = FreehandSession()
    wl = np.linspace(0.4, 0.8, 5)
    session.built = True
    session.wl_um = wl
    session.current = {"R": np.linspace(0.1, 0.2, 5), "T": np.zeros(5), "A": np.zeros(5)}
    return session, wl


def test_curve_drag_end_updates_session_target(curve_drag_session) -> None:
    session, wl = curve_drag_session
    target_r = np.linspace(0.3, 0.6, 5)
    from filmstack_simulation.filmstack_optimization.local_search.page import _handle_component_event

    triggered = _handle_component_event(
        session,
        {
            "type": "curveDragEnd",
            "target": {"R": target_r.tolist(), "T": None, "A": None},
            "touched": {"R": True, "T": False, "A": False},
            "triggerOptimize": True,
        },
    )
    assert triggered is True
    assert session.touched["R"] is True
    np.testing.assert_allclose(session.target["R"], target_r)


def test_curve_drag_end_updates_edit_wl_indices(curve_drag_session) -> None:
    session, _wl = curve_drag_session
    from filmstack_simulation.filmstack_optimization.local_search.page import _handle_component_event

    _handle_component_event(
        session,
        {
            "type": "curveDragEnd",
            "metric": "R",
            "target": {"R": np.linspace(0.3, 0.6, 5).tolist(), "T": None, "A": None},
            "touched": {"R": True, "T": False, "A": False},
            "editWlIndices": {"R": [1, 2, 3]},
            "triggerOptimize": False,
        },
    )
    assert session.edit_wl_indices["R"] == {1, 2, 3}


def test_build_freehand_config_full_scope_omits_wl_indices() -> None:
    wl = np.array([0.4, 0.5, 0.6, 0.7, 0.8])
    cfg = build_freehand_config(
        working_formula="air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0",
        wl_from=0.4,
        wl_to=0.8,
        n_wl=5,
        angle_deg=0.0,
        touched={"R": True, "T": False, "A": False},
        target={"R": np.linspace(0.1, 0.2, 5), "T": None, "A": None},
        wl_um=wl,
        view_domain={"R": {"x": [0.45, 0.55], "y": [0.0, 1.0]}},
        edit_wl_indices={"R": {1, 2}, "T": set(), "A": set()},
        cost_scope="full",
    )
    assert cfg["freehand_cost_scope"] == "full"
    assert "freehand_wl_indices" not in cfg


def test_build_freehand_config_zoom_scope_uses_view_domain() -> None:
    wl = np.array([0.4, 0.5, 0.6])
    cfg = build_freehand_config(
        working_formula="air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0",
        wl_from=0.4,
        wl_to=0.6,
        n_wl=3,
        angle_deg=0.0,
        touched={"R": True, "T": False, "A": False},
        target={"R": np.array([0.1, 0.2, 0.3]), "T": None, "A": None},
        wl_um=wl,
        view_domain={"R": {"x": [0.45, 0.55], "y": [0.0, 1.0]}},
        cost_scope="zoom",
    )
    assert cfg["freehand_cost_scope"] == "zoom"
    assert cfg["freehand_wl_indices"] == {"R": [1]}


def test_build_freehand_config_stroke_scope_uses_edit_indices() -> None:
    wl = np.linspace(0.4, 0.8, 5)
    cfg = build_freehand_config(
        working_formula="air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0",
        wl_from=0.4,
        wl_to=0.8,
        n_wl=5,
        angle_deg=0.0,
        touched={"R": True, "T": False, "A": False},
        target={"R": np.linspace(0.1, 0.2, 5), "T": None, "A": None},
        wl_um=wl,
        edit_wl_indices={"R": {1, 2}, "T": set(), "A": set()},
        cost_scope="stroke",
    )
    assert cfg["freehand_cost_scope"] == "stroke"
    assert cfg["freehand_wl_indices"] == {"R": [1, 2]}


def test_metric_active_respects_wl_index_mask() -> None:
    from filmstack_simulation.filmstack_optimization.local_search.cost_freehand import metric_active

    wl_to_idx = {0.4: 0, 0.5: 1, 0.6: 2}
    allowed = {"R": {1, 2}}
    assert metric_active("R", 0.4, wl_to_idx, allowed) is False
    assert metric_active("R", 0.5, wl_to_idx, allowed) is True
    assert metric_active("R", 0.4, wl_to_idx, {}) is True


@pytest.fixture
def freehand_cost_setup(materials_db):
    import filmstack_optimization_utils as fos

    formula = "air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0"
    spec = fos.stack_from_formula(formula, materials_db)
    wls = np.linspace(0.4, 0.8, 5)
    targets = [
        fos.SpectralTarget(
            wl_um=float(wl),
            angle_deg=0.0,
            weight=1.0,
            R_target=0.5,
        )
        for wl in wls
    ]
    x0 = np.array([spec.thicknesses_um[i] for i in spec.film_indices], dtype=float)
    return fos, spec, wls, targets, x0


def test_freehand_target_cost_uses_only_masked_indices(freehand_cost_setup) -> None:
    from filmstack_simulation.filmstack_optimization.local_search.cost_freehand import freehand_target

    fos, spec, wls, targets, x0 = freehand_cost_setup
    cfg = {
        "freehand_cost_scope": "zoom",
        "freehand_touched": {"R": True, "T": False, "A": False},
        "freehand_wl_indices": {"R": [1, 2]},
    }
    ctx = fos.make_objective_context(spec, targets, fos.Polarization.UNPOLARIZED, cfg)
    ctx.freehand_touched = cfg["freehand_touched"]
    ctx.freehand_wl_indices = cfg["freehand_wl_indices"]
    ctx.optimization_cfg = cfg

    loss_masked, _ = freehand_target(x0, ctx)
    targets[0] = fos.SpectralTarget(
        wl_um=float(wls[0]),
        angle_deg=0.0,
        weight=1.0,
        R_target=0.99,
    )
    ctx2 = fos.make_objective_context(spec, targets, fos.Polarization.UNPOLARIZED, cfg)
    ctx2.freehand_touched = cfg["freehand_touched"]
    ctx2.freehand_wl_indices = cfg["freehand_wl_indices"]
    ctx2.optimization_cfg = cfg
    loss_masked_after, _ = freehand_target(x0, ctx2)
    assert loss_masked == pytest.approx(loss_masked_after)


def test_freehand_target_full_scope_ignores_indices(freehand_cost_setup) -> None:
    from filmstack_simulation.filmstack_optimization.local_search.cost_freehand import freehand_target

    fos, spec, wls, targets, x0 = freehand_cost_setup
    cfg = {
        "freehand_cost_scope": "full",
        "freehand_touched": {"R": True, "T": False, "A": False},
        "freehand_wl_indices": {"R": [1, 2]},
    }
    ctx = fos.make_objective_context(spec, targets, fos.Polarization.UNPOLARIZED, cfg)
    ctx.freehand_touched = cfg["freehand_touched"]
    ctx.freehand_wl_indices = cfg["freehand_wl_indices"]
    ctx.optimization_cfg = cfg

    loss_full, _ = freehand_target(x0, ctx)
    targets[0] = fos.SpectralTarget(
        wl_um=float(wls[0]),
        angle_deg=0.0,
        weight=1.0,
        R_target=0.99,
    )
    ctx2 = fos.make_objective_context(spec, targets, fos.Polarization.UNPOLARIZED, cfg)
    ctx2.freehand_touched = cfg["freehand_touched"]
    ctx2.freehand_wl_indices = cfg["freehand_wl_indices"]
    ctx2.optimization_cfg = cfg
    loss_full_after, _ = freehand_target(x0, ctx2)
    assert loss_full_after > loss_full


def test_formula_from_stack_emits_unique_name(materials_db) -> None:
    air = materials_db["air_Ciddor"]
    assert formula_from_stack([air], [0.0], materials_db) == "air_Ciddor 0"


def test_formula_from_stack_emits_inline_nk_for_pseudo_material(simulation) -> None:
    h = simulation.material_s.from_nk(1.5 + 0.0j, "H")
    assert formula_from_stack([h], [0.083], {}) == "H 0.083 1.5 0"


def test_formula_from_stack_round_trip_optical_filter(materials_db, simulation) -> None:
    from filmstack_visualizer import layers_from_formula
    from filmstack_simulation.presets import build_formula_for_preset
    from toykits_config import FILMSTACK_PRESET_CATALOG

    formula = build_formula_for_preset("optical_filter", FILMSTACK_PRESET_CATALOG)
    mats, th = layers_from_formula(formula, materials_db, simulation_module=simulation)
    round_trip = formula_from_stack(mats, th, materials_db)
    mats2, th2 = layers_from_formula(round_trip, materials_db, simulation_module=simulation)
    assert len(mats2) == len(mats)
    assert len(th2) == len(th)


def test_layers_from_formula_resolves_unique_air_name(materials_db, simulation) -> None:
    from filmstack_visualizer import layers_from_formula

    unique_air = "air_Ciddor"
    formula = f"{unique_air} 0 SiO2_Arosa 0.1 Si_Aspnes 0"
    mats, _th = layers_from_formula(formula, materials_db, simulation_module=simulation)
    assert len(mats) >= 2


def test_build_freehand_config_touched_r_only() -> None:
    target_r = np.linspace(0.1, 0.2, 5)
    cfg = build_freehand_config(
        working_formula="Vacuum 0 1 0 SiO2_Arosa 0.1 Si_Aspnes 0",
        wl_from=0.4,
        wl_to=0.8,
        n_wl=5,
        angle_deg=0.0,
        touched={"R": True, "T": False, "A": False},
        target={"R": target_r, "T": None, "A": None},
    )
    assert cfg["cost_function"]["name"] == "freehand_target"
    assert cfg["R_target_spectrum"] == [target_r.tolist()]
    assert "T_target_spectrum" not in cfg
    assert cfg["freehand_touched"]["R"] is True


def test_freehand_session_reset_clears_targets() -> None:
    session = FreehandSession()
    session.touched["R"] = True
    session.target["R"] = np.array([0.1, 0.2])
    session.last_optimized_formula = "air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0"
    wl = np.linspace(0.4, 0.8, 2)
    session.reset_after_build(
        formula="test",
        wl_um=wl,
        angle_deg=0.0,
        current={"R": wl * 0, "T": wl * 0 + 0.5, "A": wl * 0 + 0.5},
        wl_from=0.4,
        wl_to=0.8,
    )
    assert session.touched["R"] is False
    assert session.target["R"] is None
    assert session.edit_wl_indices["R"] == set()
    assert session.opt_round == 0
    assert session.baseline_formula == "test"
    assert session.last_optimized_formula is None


def test_apply_optimization_result_preserves_baseline() -> None:
    session = FreehandSession()
    wl = np.linspace(0.4, 0.8, 3)
    session.reset_after_build(
        formula="air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0",
        wl_um=wl,
        angle_deg=0.0,
        current={"R": wl * 0.1, "T": wl * 0.5, "A": wl * 0.4},
        wl_from=0.4,
        wl_to=0.8,
    )
    session.apply_optimization_result(
        formula="air_Ciddor 0 SiO2_Arosa 0.12 Si_Aspnes 0",
        current={"R": wl * 0.12, "T": wl * 0.48, "A": wl * 0.4},
        merit_history=[0.1, 0.05],
        merit_initial=0.1,
    )
    assert session.baseline_formula == "air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0"
    assert session.working_formula == "air_Ciddor 0 SiO2_Arosa 0.12 Si_Aspnes 0"
    assert session.last_optimized_formula == "air_Ciddor 0 SiO2_Arosa 0.12 Si_Aspnes 0"
    assert session.opt_round == 1


def test_combine_polarization_rt_te_tm_unpolarized() -> None:
    r_s = np.array([0.2, 0.4])
    t_s = np.array([0.7, 0.5])
    r_p = np.array([0.6, 0.8])
    t_p = np.array([0.3, 0.1])
    r_te, t_te = combine_polarization_rt(r_s, t_s, r_p, t_p, "TE")
    r_tm, t_tm = combine_polarization_rt(r_s, t_s, r_p, t_p, "TM")
    r_u, t_u = combine_polarization_rt(r_s, t_s, r_p, t_p, "UNPOLARIZED")
    assert np.allclose(r_te, r_s)
    assert np.allclose(t_te, t_s)
    assert np.allclose(r_tm, r_p)
    assert np.allclose(t_tm, t_p)
    assert np.allclose(r_u, 0.5 * (r_s + r_p))
    assert np.allclose(t_u, 0.5 * (t_s + t_p))


def test_build_freehand_config_includes_polarization() -> None:
    cfg = build_freehand_config(
        working_formula="air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0",
        wl_from=0.4,
        wl_to=0.8,
        n_wl=5,
        angle_deg=0.0,
        touched={"R": False, "T": False, "A": False},
        target={"R": None, "T": None, "A": None},
        polarization="TE",
    )
    assert cfg["polarization"] == "TE"


def test_compute_rta_at_angle_polarization_modes(polarization_test_stack) -> None:
    materials, thicknesses_um = polarization_test_stack
    kwargs = {
        "materials": materials,
        "thicknesses_um": thicknesses_um,
        "angle_deg": 30.0,
        "wl_from": 0.45,
        "wl_to": 0.75,
        "n_wl": 8,
    }
    unpol = compute_rta_at_angle(**kwargs, polarization="UNPOLARIZED")
    te = compute_rta_at_angle(**kwargs, polarization="TE")
    tm = compute_rta_at_angle(**kwargs, polarization="TM")
    assert unpol["R"].shape == te["R"].shape == tm["R"].shape
    assert not np.allclose(te["R"], tm["R"])
    assert np.allclose(unpol["A"], 1.0 - unpol["R"] - unpol["T"])


def test_compute_spectral_map_2d_polarization_modes(polarization_test_stack) -> None:
    from filmstack_simulation.simulation import compute_spectral_map_2d

    materials, thicknesses_um = polarization_test_stack
    kwargs = {
        "materials": materials,
        "thicknesses_um": thicknesses_um,
        "wl_from": 0.45,
        "wl_to": 0.75,
        "ang_from": 0.0,
        "ang_to": 30.0,
        "n_wl": 4,
        "n_ang": 4,
    }
    unpol = compute_spectral_map_2d(**kwargs, polarization="UNPOLARIZED")
    te = compute_spectral_map_2d(**kwargs, polarization="TE")
    tm = compute_spectral_map_2d(**kwargs, polarization="TM")
    assert unpol["R"].shape == te["R"].shape == tm["R"].shape
    assert not np.allclose(te["R"], tm["R"])
    assert np.array_equal(unpol["Psi"], te["Psi"])
    assert np.array_equal(unpol["Delta"], te["Delta"])


def test_film_layer_indices() -> None:
    from filmstack_simulation.filmstack_optimization.shared.stack_table import film_layer_indices

    assert film_layer_indices(3) == [1]
    assert film_layer_indices(9) == list(range(1, 8))


def test_layer_bounds_from_ranges() -> None:
    from filmstack_simulation.filmstack_optimization.shared.stack_table import layer_bounds_from_ranges

    t0 = 0.1
    bounds = layer_bounds_from_ranges([1], [0.0, t0, 0.0], {1: 30.0})
    assert bounds == [{"index": 1, "min": pytest.approx(0.07), "max": pytest.approx(0.13)}]

    fixed = layer_bounds_from_ranges([1], [0.0, t0, 0.0], {1: 0.0})
    assert fixed[0]["min"] == pytest.approx(t0)
    assert fixed[0]["max"] == pytest.approx(t0)

    wide = layer_bounds_from_ranges([1], [0.0, t0, 0.0], {1: 100.0})
    assert wide[0]["min"] == pytest.approx(0.0)
    assert wide[0]["max"] == pytest.approx(2 * t0)


def test_stack_table_rows_includes_range_column() -> None:
    import pandas as pd
    from filmstack_simulation.filmstack_optimization.shared.stack_table import stack_table_rows

    df = stack_table_rows(
        ["air", "SiO2", "Si"],
        [0.0, 0.1, 0.0],
        layer_range_pct={1: 30.0},
        film_indices=[1],
    )
    assert list(df.columns) == ["_idx", "材料", "厚度 (μm)", "厚度变化范围 (%)"]
    assert df.at[0, "厚度 (μm)"] == "∞"
    assert pd.isna(df.at[0, "厚度变化范围 (%)"])
    assert df.at[1, "厚度变化范围 (%)"] == pytest.approx(30.0)
    assert pd.isna(df.at[2, "厚度变化范围 (%)"])


def test_sync_layer_range_pct_from_table() -> None:
    import pandas as pd
    from filmstack_simulation.filmstack_optimization.shared.stack_table import sync_layer_range_pct_from_table

    edited = pd.DataFrame(
        {
            "_idx": [0, 1, 2],
            "材料": ["air", "SiO2", "Si"],
            "厚度 (μm)": ["∞", "0.10000", "∞"],
            "厚度变化范围 (%)": [pd.NA, 25.0, pd.NA],
        }
    )
    assert sync_layer_range_pct_from_table(edited, [1]) == {1: 25.0}


def test_sync_layer_range_pct_uses_idx_after_reorder() -> None:
    import pandas as pd
    from filmstack_simulation.filmstack_optimization.shared.stack_table import sync_layer_range_pct_from_table

    edited = pd.DataFrame(
        {
            "_idx": [2, 0, 1],
            "材料": ["Si", "air", "SiO2"],
            "厚度 (μm)": ["∞", "∞", "0.10000"],
            "厚度变化范围 (%)": [pd.NA, pd.NA, 40.0],
        }
    )
    assert sync_layer_range_pct_from_table(edited, [1]) == {1: 40.0}


def test_reset_after_build_initializes_layer_range_pct() -> None:
    session = FreehandSession()
    wl = np.linspace(0.4, 0.8, 3)
    session.reset_after_build(
        formula="air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0",
        wl_um=wl,
        angle_deg=0.0,
        current={"R": np.zeros(3), "T": np.zeros(3), "A": np.zeros(3)},
        wl_from=0.4,
        wl_to=0.8,
        film_indices=[1],
        default_range_pct=30.0,
    )
    assert session.layer_range_pct == {1: 30.0}


def test_apply_optimization_result_preserves_layer_range_pct() -> None:
    session = FreehandSession()
    wl = np.linspace(0.4, 0.8, 3)
    session.reset_after_build(
        formula="air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0",
        wl_um=wl,
        angle_deg=0.0,
        current={"R": np.zeros(3), "T": np.zeros(3), "A": np.zeros(3)},
        wl_from=0.4,
        wl_to=0.8,
        film_indices=[1],
        default_range_pct=30.0,
    )
    session.layer_range_pct[1] = 15.0
    session.apply_optimization_result(
        formula="air_Ciddor 0 SiO2_Arosa 0.12 Si_Aspnes 0",
        current={"R": np.zeros(3), "T": np.zeros(3), "A": np.zeros(3)},
        merit_history=[1.0, 0.5],
    )
    assert session.layer_range_pct == {1: 15.0}


def test_build_freehand_config_includes_layer_bounds() -> None:
    import filmstack_optimization_utils as fos  # noqa: E402

    cfg = build_freehand_config(
        working_formula="air_Ciddor 0 SiO2_Arosa 0.1 Si_Aspnes 0",
        wl_from=0.4,
        wl_to=0.8,
        n_wl=5,
        angle_deg=0.0,
        touched={"R": False, "T": False, "A": False},
        target={"R": None, "T": None, "A": None},
        film_indices=[1],
        thicknesses_um=[0.0, 0.1, 0.0],
        layer_range_pct={1: 30.0},
    )
    assert cfg["layer_bounds"] == [
        {"index": 1, "min": pytest.approx(0.07), "max": pytest.approx(0.13)}
    ]
    assert fos.resolve_layer_bounds(cfg, [1]) == [
        (pytest.approx(0.07), pytest.approx(0.13))
    ]


def test_get_freehand_default_thickness_range_pct() -> None:
    from filmstack_simulation.filmstack_optimization.local_search.opt_config import (
        get_freehand_default_thickness_range_pct,
    )

    assert get_freehand_default_thickness_range_pct() == pytest.approx(30.0)


def test_clamp_freehand_target_array() -> None:
    from filmstack_simulation.filmstack_optimization.local_search.freehand_state import (
        clamp_freehand_target_array,
    )

    arr = np.array([0.0, 0.5, 1.2, -0.1])
    clamped = clamp_freehand_target_array(arr)
    np.testing.assert_allclose(clamped, [0.0, 0.5, 1.0, 0.0])


def test_validate_freehand_targets_accepts_unit_interval() -> None:
    from filmstack_simulation.filmstack_optimization.local_search.freehand_state import (
        validate_freehand_targets,
    )

    validate_freehand_targets(
        {"R": True, "T": False, "A": False},
        {"R": np.linspace(0.0, 1.0, 5), "T": None, "A": None},
    )


def test_validate_freehand_targets_rejects_out_of_range() -> None:
    from filmstack_simulation.filmstack_optimization.local_search.freehand_state import (
        validate_freehand_targets,
    )

    with pytest.raises(ValueError, match="R 目标值须在"):
        validate_freehand_targets(
            {"R": True, "T": False, "A": False},
            {"R": np.array([0.1, 1.5]), "T": None, "A": None},
        )


def test_validate_freehand_targets_rejects_nan() -> None:
    from filmstack_simulation.filmstack_optimization.local_search.freehand_state import (
        validate_freehand_targets,
    )

    with pytest.raises(ValueError, match="R 目标值须为有限数值"):
        validate_freehand_targets(
            {"R": True, "T": False, "A": False},
            {"R": np.array([0.1, np.nan]), "T": None, "A": None},
        )


def test_handle_component_event_clamps_target(curve_drag_session) -> None:
    session, _wl = curve_drag_session
    from filmstack_simulation.filmstack_optimization.local_search.page import _handle_component_event

    _handle_component_event(
        session,
        {
            "type": "curveDragEnd",
            "target": {"R": [0.3, 1.2], "T": None, "A": None},
            "touched": {"R": True, "T": False, "A": False},
            "triggerOptimize": False,
        },
    )
    np.testing.assert_allclose(session.target["R"], [0.3, 1.0])


def test_apply_optimized_thicknesses_preserves_mg_and_incoherent() -> None:
    from filmstack_simulation.filmstack_optimization.shared.stack_table import (
        apply_optimized_thicknesses_to_formula,
    )

    orig = "air 0 1.0 0.0 [8e-06 au glass 1.5] 4000* air 0 1.0 0.0"
    out = apply_optimized_thicknesses_to_formula(orig, [1], [0.0, 3999.92903, 0.0])
    assert "[8e-06 au glass 1.5]" in out
    assert "3999.92903*" in out
    assert "MG(" not in out
    assert "1.0 0.0" in out


def test_apply_optimized_thicknesses_bookend_stack(simulation) -> None:
    from filmstack_simulation.filmstack_optimization.shared.stack_table import (
        apply_optimized_thicknesses_to_formula,
        film_layer_indices,
    )
    from filmstack_visualizer import layers_from_formula_with_flags

    orig = "air 0.1 SiO2 0.1 1.46 0 Si 0.1 3.87 0.02"
    _, th, _ = layers_from_formula_with_flags(orig, {}, simulation_module=simulation)
    film_indices = film_layer_indices(len(th))
    new_th = list(th)
    new_th[2] = 0.12
    out = apply_optimized_thicknesses_to_formula(orig, film_indices, new_th)
    assert "SiO2 0.12" in out
    assert "1.46 0" in out


def test_apply_optimized_thicknesses_round_trip_incoherent(simulation) -> None:
    from filmstack_simulation.filmstack_optimization.shared.stack_table import (
        apply_optimized_thicknesses_to_formula,
        film_layer_indices,
    )
    from filmstack_visualizer import layers_from_formula_with_flags

    orig = "air 0 film 100* 1.5 0 air 0"
    _, th, flags = layers_from_formula_with_flags(orig, {}, simulation_module=simulation)
    film_indices = film_layer_indices(len(th))
    new_th = list(th)
    new_th[1] = 95.5
    out = apply_optimized_thicknesses_to_formula(orig, film_indices, new_th)
    _, _, flags2 = layers_from_formula_with_flags(out, {}, simulation_module=simulation)
    assert flags2[1] is True
    assert "95.5*" in out


def test_layers_has_incoherent_on_star_formula(simulation) -> None:
    from filmstack_visualizer import build_tmm_layers, layers_from_formula_with_flags, layers_has_incoherent

    formula = "air 0 film 100* 1.5 0 air 0"
    mats, th, flags = layers_from_formula_with_flags(formula, {}, simulation_module=simulation)
    layers = build_tmm_layers(mats, th, incoherent_flags=flags, simulation_module=simulation)
    assert layers_has_incoherent(layers) is True
