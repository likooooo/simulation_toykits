"""simulation_database.plots 测试用例"""

import numpy as np
import pytest

pytest.importorskip("plotly")

from simulation_database.plots import COLORS, build_nk_curve_figure, build_spectrum_curve_figure


class TestBuildNkCurveFigure:
    def test_returns_figure(self):
        wls = [0.4, 0.6, 0.8]
        n_vals = [1.45, 1.46, 1.44]
        k_vals = [0.0, 0.0, 0.001]
        fig = build_nk_curve_figure(wls, n_vals, k_vals, title="Test")
        assert fig is not None
        assert hasattr(fig, "data")
        assert len(fig.data) == 2
        assert fig.layout.yaxis.title.text == "n"
        assert getattr(fig.layout, "yaxis2", None) is not None
        assert fig.layout.yaxis2.title.text == "k"
        assert fig.data[0].line.color == COLORS["material_n"]
        assert fig.data[1].line.color == COLORS["material_k"]

    def test_numpy_arrays(self):
        wls = np.linspace(0.4, 0.8, 10)
        n_vals = np.ones(10) * 1.5
        k_vals = np.zeros(10)
        fig = build_nk_curve_figure(wls, n_vals, k_vals)
        assert len(fig.data[0].x) == 10
        assert len(fig.data[0].y) == 10

    def test_spectrum_theme_no_yaxis2_error(self):
        fig = build_spectrum_curve_figure([0.4, 0.6], [1.0, 2.0], title="Spec", height=300)
        assert getattr(fig.layout, "yaxis2", None) is None

    def test_sim_wl_markers_uniform_style(self):
        wls = [0.3, 0.5, 0.9]
        n_vals = [1.45, 1.46, 1.44]
        k_vals = [0.0, 0.0, 0.001]
        fig = build_nk_curve_figure(
            wls, n_vals, k_vals, sim_wl_from=0.35, sim_wl_to=0.8
        )
        marker_shapes = [
            s for s in fig.layout.shapes if s.type == "line" and s.xref == "x"
        ]
        assert len(marker_shapes) == 2
        styles = {
            (s.line.color, s.line.width, s.line.dash, s.layer) for s in marker_shapes
        }
        assert len(styles) == 1
        color, width, dash, layer = next(iter(styles))
        assert color == COLORS["text"]
        assert width == 2.5
        assert dash == "dash"
        assert layer == "above"

    def test_x_axis_padded_by_wavelength_span(self):
        wls = [0.3, 0.5, 0.9]
        n_vals = [1.45, 1.46, 1.44]
        k_vals = [0.0, 0.0, 0.001]
        fig = build_nk_curve_figure(
            wls, n_vals, k_vals, sim_wl_from=0.35, sim_wl_to=0.8
        )
        wl_min = min(0.3, 0.35)
        wl_max = max(0.9, 0.8)
        assert fig.layout.xaxis.range == pytest.approx([wl_min * 0.9, wl_max * 1.1])
