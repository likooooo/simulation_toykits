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
