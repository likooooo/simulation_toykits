"""Tests for core.beams (compute) and core.beams_plot (show_complex_plot)."""

import numpy as np
import pytest

from core.beams_plot import show_complex_plot

_GRID = ([-1.0, -1.0], [1.0, 1.0], [4, 4])


class TestShowComplexPlot:
    """Test show_complex_plot without simulation.so."""

    def test_returns_fig_with_four_subplots(self):
        arr = np.exp(1j * np.linspace(0, 2 * np.pi, 16)).reshape(4, 4)
        meta = {"nx": 4, "ny": 4, "dx": 1.0, "dy": 1.0, "wavelength": 0.5}
        fig = show_complex_plot(arr, meta, title_prefix="test")
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) >= 4
        titles = [ax.get_title() for ax in axes[:4]]
        assert "Real Part" in titles
        assert "Imaginary Part" in titles
        assert "Amplitude" in titles
        assert "Phase (rad)" in titles

    def test_meta_without_dx_dy_uses_pixel_labels(self):
        arr = np.zeros((3, 3), dtype=complex)
        meta = {"nx": 3, "ny": 3}
        fig = show_complex_plot(arr, meta)
        axes = fig.get_axes()
        assert any(ax.get_xlabel() == "x (px)" for ax in axes)


def _assert_beam_shape(field, meta) -> None:
    assert field.shape == (4, 4)
    assert meta["nx"] == 4 and meta["ny"] == 4
    assert "dx" in meta and "dy" in meta and "wavelength" in meta


class TestBeamsCompute:
    """Smoke tests for beam compute functions; 依赖 simulation.so，加载失败则用例直接失败并报真实错误。"""

    @pytest.mark.parametrize(
        "import_path,args",
        [
            ("compute_plane_wave", (0.5, 0.0, 0.0)),
            ("compute_quadratic_wave", (0.5, 1.0)),
            ("compute_spherical_wave", (0.5, 1.0)),
            (
                "compute_flat_top_rectangular",
                (1.0, 1.0, 0.5, 2.0, 2.0),
            ),
            ("compute_flat_top_circular", (1.0, 0.5, 2.0)),
            ("compute_hermite_gaussian", (0, 0, 0.5, 0.0, 1.0, 1.0)),
            ("compute_laguerre_gaussian", (0, 0, 0.5, 0.0, 1.0)),
        ],
    )
    def test_beam_compute_smoke(self, import_path, args):
        import importlib

        mod = importlib.import_module("core.beams")
        compute_fn = getattr(mod, import_path)
        start_xy, end_xy, shape_xy = _GRID
        field, meta = compute_fn(*args, start_xy, end_xy, shape_xy)
        _assert_beam_shape(field, meta)
