"""filmstack_visualizer formula parsing, layers_from_formula, and build_tmm_layers tests."""

from __future__ import annotations

import pytest

from filmstack_visualizer import (
    build_tmm_layers,
    layers_from_formula,
    normalize_filmstack_formula,
    parse_filmstack_formula_v1,
    plot_filmstack,
)


def _plot_filmstack_texts(fig) -> list[str]:
    return [t.get_text() for t in fig.axes[0].texts]


def _plot_filmstack_legend_texts(fig) -> list[str]:
    legend = fig.axes[0].get_legend()
    assert legend is not None
    return [t.get_text() for t in legend.get_texts()]


def _plot_filmstack_bar_count(fig) -> int:
    return len(fig.axes[0].patches)


def _plot_filmstack_zero_vlines(fig) -> list:
    ax = fig.axes[0]
    return [line for line in ax.lines if line.get_zorder() == 5]


class TestParseFormula:
    def test_vacuum_sio2(self):
        got = parse_filmstack_formula_v1("Vacuum 0 SiO2 0.5")
        assert len(got) == 2
        assert got[0]["Material"] == "Vacuum"
        assert got[1]["Thickness (um)"] == 0.5

    def test_inline_nk(self):
        got = parse_filmstack_formula_v1("SiO2 0.1 1.46 0.001")
        assert got[0]["n"] == 1.46
        assert got[0]["k"] == 0.001

    def test_periodic(self):
        got = parse_filmstack_formula_v1("(SiO2 0.1 Ta2O5 0.02)^2")
        assert len(got) == 4

    def test_whitespace(self):
        got = parse_filmstack_formula_v1("  Vacuum 0   SiO2 0.1  ")
        assert got[0]["Material"] == "Vacuum"
        assert got[1]["Material"] == "SiO2"

    def test_multi_inline_nk(self):
        got = parse_filmstack_formula_v1("A 0.1 1.5 0.0 B 0.2 C 0.05 2.0 0.001")
        assert got[0]["n"] == 1.5
        assert got[2]["n"] == 2.0

    def test_bragg_stack(self):
        got = parse_filmstack_formula_v1("(H 0.01 L 0.02)^5")
        assert len(got) == 10

    def test_malformed_incomplete_layer(self):
        with pytest.raises(ValueError, match="incomplete layer"):
            parse_filmstack_formula_v1("SiO2 0.1 SiO2")

    def test_normalize_newlines(self):
        raw = "air 0\n(SiO2 0.1 Ta2O5 0.02)^2\nSi 0"
        flat = "air 0 (SiO2 0.1 Ta2O5 0.02)^2 Si 0"
        assert normalize_filmstack_formula(raw) == flat
        assert parse_filmstack_formula_v1(normalize_filmstack_formula(raw)) == parse_filmstack_formula_v1(flat)


class TestLayersFromFormula:
    def test_resolve_strips_newlines(self, simulation):
        raw = "air 0\nSiO2 0.05 1.46 0.0\nSi 0 3.87 0.02"
        flat = "air 0 SiO2 0.05 1.46 0.0 Si 0 3.87 0.02"
        mats_raw, th_raw = layers_from_formula(raw, {}, simulation_module=simulation)
        mats_flat, th_flat = layers_from_formula(flat, {}, simulation_module=simulation)
        assert len(mats_raw) == len(mats_flat)
        assert th_raw == th_flat

    def test_standard_bookends_unchanged(self, simulation):
        _, th = layers_from_formula("air 0 SiO2 0.1 1.46 0 Si 0 3.87 0.02", {}, simulation_module=simulation)
        assert th[0] == 0.0
        assert th[-1] == 0.0
        assert th == [0.0, 0.1, 0.0]

    def test_both_ends_expanded(self, simulation):
        _, th = layers_from_formula(
            "air 0.1 SiO2 0.1 1.46 0 Si 0.1 3.87 0.02",
            {},
            simulation_module=simulation,
        )
        assert th[0] == 0.0
        assert th[-1] == 0.0
        assert th == [0.0, 0.1, 0.1, 0.1, 0.0]

    def test_prepend_only(self, simulation):
        _, th = layers_from_formula("air 0.1 SiO2 0.1 1.46 0 Si 0 3.87 0.02", {}, simulation_module=simulation)
        assert th == [0.0, 0.1, 0.1, 0.0]

    def test_append_only(self, simulation):
        _, th = layers_from_formula("air 0 SiO2 0.1 1.46 0 Si 0.1 3.87 0.02", {}, simulation_module=simulation)
        assert th == [0.0, 0.1, 0.1, 0.0]


class TestBuildTmmLayers:
    def test_attach_layers_from_formula(self, simulation):
        materials, thicknesses = layers_from_formula(
            "air 0 SiO2 0.1 1.46 0 Si 0 3.87 0.02",
            {},
            simulation_module=simulation,
        )
        layers = build_tmm_layers(materials, thicknesses, simulation_module=simulation)
        assert len(layers) == len(materials)
        assert float(layers[0].depth) == 0.0
        assert float(layers[-1].depth) == 0.0
        assert float(layers[1].depth) == pytest.approx(0.1)

    def test_preserves_formula_material_names(self, simulation):
        formula = "air 0 (H 0.1 1.5 0 air 0.1)^30 Si 0 3.87 0.02"
        mats, th = layers_from_formula(formula, {}, simulation_module=simulation)
        layers = build_tmm_layers(mats, th, simulation_module=simulation)
        names = [lyr.background_material.name for lyr in layers]
        assert "upper" not in names
        assert "substrate" not in names
        assert not any(n.startswith("layer_") for n in names)
        assert names[0] == "air"
        assert names[-1] == "Si"
        assert set(names[1:-1]) == {"air", "H"}


class TestPlotFilmstackMaterialLabelsRegression:
    """Regression: plot_filmstack legend/halfspace labels use formula material names.

    Bug: build_tmm_layers renamed materials to layer_N/upper/substrate, yielding
    60 legend entries for Bragg ^30 stacks and generic halfspace labels.
    """

    def test_rejects_non_bookend_input(self, simulation):
        mats, th = layers_from_formula("air 0 Si 0 3.87 0.02", {}, simulation_module=simulation)
        layers = build_tmm_layers(mats, th, simulation_module=simulation)
        layers[0].depth = 0.1
        with pytest.raises(ValueError, match="bookend"):
            plot_filmstack(layers)

    def test_margin_infinity_labels(self, simulation):
        import matplotlib

        matplotlib.use("Agg")
        mats, th = layers_from_formula("air 0.1 Si 0.1 3.87 0.02", {}, simulation_module=simulation)
        layers = build_tmm_layers(mats, th, simulation_module=simulation)
        fig = plot_filmstack(layers)
        infty_labels = [t for t in _plot_filmstack_texts(fig) if "(∞)" in t]
        assert infty_labels == ["air(∞)", "Si(∞)"]

    def test_legend_groups_by_material_name(self, simulation):
        import matplotlib

        matplotlib.use("Agg")
        formula = "air 0 (H 0.1 1.5 0 air 0.1)^5 Si 0 3.87 0.02"
        mats, th = layers_from_formula(formula, {}, simulation_module=simulation)
        layers = build_tmm_layers(mats, th, simulation_module=simulation)
        fig = plot_filmstack(layers, layer_label_mode="legend")
        assert {t for t in _plot_filmstack_legend_texts(fig)} == {"air", "H"}

    def test_bragg_30_legend_count_and_no_layer_aliases(self, simulation):
        import matplotlib

        matplotlib.use("Agg")
        formula = "air 0 (H 0.09598 1.5 0.0 air 0.11302)^30 Si 0 3.87 0.02"
        mats, th = layers_from_formula(formula, {}, simulation_module=simulation)
        layers = build_tmm_layers(mats, th, simulation_module=simulation)
        fig = plot_filmstack(layers, layer_label_mode="legend")
        legend_texts = _plot_filmstack_legend_texts(fig)
        assert len(legend_texts) == 2
        assert set(legend_texts) == {"air", "H"}
        assert not any(t.startswith("layer_") for t in legend_texts)

        all_text = _plot_filmstack_texts(fig)
        assert not any("upper" in t or "substrate" in t for t in all_text)
        assert "air(∞)" in all_text and "Si(∞)" in all_text

    def test_zero_thickness_middle_layers_drawn(self, simulation):
        import matplotlib

        matplotlib.use("Agg")
        formula = "air 0 (SiO2 0.1 1.46 0 Ta2O5 0 2.13 0.001)^2 Si 0 3.87 0.02"
        mats, th = layers_from_formula(formula, {}, simulation_module=simulation)
        layers = build_tmm_layers(mats, th, simulation_module=simulation)
        fig = plot_filmstack(layers, layer_label_mode="legend")
        legend_texts = _plot_filmstack_legend_texts(fig)
        assert {"SiO2", "Ta2O5"}.issubset(set(legend_texts))
        assert _plot_filmstack_bar_count(fig) == 2
        assert len(_plot_filmstack_zero_vlines(fig)) == 2
        all_text = _plot_filmstack_texts(fig)
        assert "air(∞)" in all_text and "Si(∞)" in all_text

    def test_zero_thickness_vline_width_uniform(self, simulation):
        import matplotlib

        matplotlib.use("Agg")
        formula = "air 0 (SiO2 0.1 1.46 0 Ta2O5 0 2.13 0.001)^5 Si 0 3.87 0.02"
        mats, th = layers_from_formula(formula, {}, simulation_module=simulation)
        layers = build_tmm_layers(mats, th, simulation_module=simulation)
        fig = plot_filmstack(layers, layer_label_mode="legend")
        zero_vlines = _plot_filmstack_zero_vlines(fig)
        assert len(zero_vlines) == 5
        linewidths = {line.get_linewidth() for line in zero_vlines}
        assert len(linewidths) == 1
