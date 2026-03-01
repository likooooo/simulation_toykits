"""core.filmstack_viz 测试用例"""

import numpy as np
import pytest
from core.filmstack_viz import (
    MockMaterial,
    MockFilm,
    calculate_angles,
    nk_to_color,
    build_layers_for_visualization,
)


class TestCalculateAngles:
    def test_snell(self):
        layers = [
            MockFilm(MockMaterial("air", 1.0 + 0j), float("inf")),
            MockFilm(MockMaterial("glass", 1.5 + 0j), 0.1),
            MockFilm(MockMaterial("air", 1.0 + 0j), float("inf")),
        ]
        th0 = np.radians(30)
        angles = calculate_angles(layers, th0)
        assert len(angles) == 3
        assert np.degrees(np.real(angles[0])) == pytest.approx(30)
        # n1 sin(th1) = n2 sin(th2) => sin(th2) = 1*sin(30)/1.5
        assert np.degrees(np.real(angles[1])) == pytest.approx(
            np.degrees(np.arcsin(0.5 / 1.5)), rel=1e-5
        )


class TestNkToColor:
    def test_returns_rgb_tuple(self):
        r, g, b = nk_to_color(1.5, 0.0)
        assert 0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1


class TestBuildLayersForVisualization:
    def test_first_last_infinite(self):
        names = ["Air", "Film", "Sub"]
        nk_list = [1.0 + 0j, 1.5 + 0j, 1.0 + 0j]
        thickness_list = [0.0, 0.1, 0.0]
        layers = build_layers_for_visualization(names, nk_list, thickness_list)
        assert len(layers) == 3
        assert layers[0].d == float("inf")
        assert layers[1].d == 0.1
        assert layers[2].d == float("inf")
        assert layers[0].name == "Air"
        assert layers[1].nk == 1.5 + 0j
