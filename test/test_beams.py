"""Tests for core.beams (compute) and core.beams_plot (show_complex_plot)."""

import numpy as np
import pytest

from core.beams_plot import show_complex_plot


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


def _simulation_available():
    try:
        from core import simulation_loader
        simulation_loader.get_simulation_module()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _simulation_available(), reason="simulation.so not available (e.g. CI)")
class TestBeamsCompute:
    """Smoke tests for beam compute functions when simulation.so is present."""

    @staticmethod
    def _common_shape_meta(field, meta):
        assert field.shape == (4, 4)
        assert meta["nx"] == 4 and meta["ny"] == 4
        assert "dx" in meta and "dy" in meta and "wavelength" in meta

    def test_compute_plane_wave(self):
        from core.beams import compute_plane_wave
        field, meta = compute_plane_wave(
            0.5, 0.0, 0.0, [-1.0, -1.0], [1.0, 1.0], [4, 4]
        )
        self._common_shape_meta(field, meta)

    def test_compute_quadratic_wave(self):
        from core.beams import compute_quadratic_wave
        field, meta = compute_quadratic_wave(
            0.5, 1.0, [-1.0, -1.0], [1.0, 1.0], [4, 4]
        )
        self._common_shape_meta(field, meta)

    def test_compute_spherical_wave(self):
        from core.beams import compute_spherical_wave
        field, meta = compute_spherical_wave(
            0.5, 1.0, [-1.0, -1.0], [1.0, 1.0], [4, 4]
        )
        self._common_shape_meta(field, meta)

    def test_compute_flat_top_rectangular(self):
        from core.beams import compute_flat_top_rectangular
        field, meta = compute_flat_top_rectangular(
            1.0, 1.0, 0.5, 2.0, 2.0, [-1.0, -1.0], [1.0, 1.0], [4, 4]
        )
        self._common_shape_meta(field, meta)

    def test_compute_flat_top_circular(self):
        from core.beams import compute_flat_top_circular
        field, meta = compute_flat_top_circular(
            1.0, 0.5, 2.0, [-1.0, -1.0], [1.0, 1.0], [4, 4]
        )
        self._common_shape_meta(field, meta)

    def test_compute_hermite_gaussian(self):
        from core.beams import compute_hermite_gaussian
        field, meta = compute_hermite_gaussian(
            0, 0, 0.5, 0.0, 1.0, 1.0, [-1.0, -1.0], [1.0, 1.0], [4, 4]
        )
        self._common_shape_meta(field, meta)

    def test_compute_laguerre_gaussian(self):
        from core.beams import compute_laguerre_gaussian
        field, meta = compute_laguerre_gaussian(
            0, 0, 0.5, 0.0, 1.0, [-1.0, -1.0], [1.0, 1.0], [4, 4]
        )
        self._common_shape_meta(field, meta)
