"""core.plots 测试用例"""

import numpy as np
import pytest

pytest.importorskip("plotly")

from core.plots import build_nk_curve_figure


class TestBuildNkCurveFigure:
    def test_returns_figure(self):
        wls = [0.4, 0.6, 0.8]
        n_vals = [1.45, 1.46, 1.44]
        k_vals = [0.0, 0.0, 0.001]
        fig = build_nk_curve_figure(wls, n_vals, k_vals, title="Test")
        assert fig is not None
        assert hasattr(fig, "data")
        assert len(fig.data) == 2

    def test_numpy_arrays(self):
        wls = np.linspace(0.4, 0.8, 10)
        n_vals = np.ones(10) * 1.5
        k_vals = np.zeros(10)
        fig = build_nk_curve_figure(wls, n_vals, k_vals)
        assert len(fig.data[0].x) == 10
        assert len(fig.data[0].y) == 10
