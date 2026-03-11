"""Smoke tests for core.sturm_liouville (load_mat_v7, formula, run_sl, build_result_mat, plot)."""

import io
import os

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from core import sturm_liouville


class TestLoadMatV7:
    def test_load_from_bytes(self):
        buf = io.BytesIO()
        u = np.zeros((8, 8))
        f = np.ones((8, 8))
        savemat(buf, {"u": u, "f": f}, format="5", do_compression=False)
        data = sturm_liouville.load_mat_v7(buf.getvalue())
        assert "u" in data and "f" in data
        assert data["u"].shape == (8, 8)
        assert data["f"].shape == (8, 8)

    def test_load_rejects_dimension_mismatch(self):
        buf = io.BytesIO()
        savemat(buf, {"u": np.zeros((4, 4)), "f": np.zeros((3, 3))}, format="5", do_compression=False)
        with pytest.raises(ValueError, match="维度必须一致"):
            sturm_liouville.load_mat_v7(buf.getvalue())

    def test_load_requires_at_least_one_variable(self):
        buf = io.BytesIO()
        savemat(buf, {"x": np.zeros((2, 2))}, format="5", do_compression=False)
        with pytest.raises(ValueError, match="未找到变量"):
            sturm_liouville.load_mat_v7(buf.getvalue())


class TestSlFormulaMarkdown:
    def test_empty_axes(self):
        out = sturm_liouville.sl_formula_markdown([])
        assert "请添加坐标轴" in out

    def test_single_axis_p1_q0(self):
        out = sturm_liouville.sl_formula_markdown([
            {"p": 1.0, "q": 0.0},
        ])
        assert "partial" in out and "= f" in out
        assert "x_1" in out

    def test_two_axes_with_coefficients(self):
        out = sturm_liouville.sl_formula_markdown([
            {"p": 1.0, "q": 0.0},
            {"p": 2.0, "q": 1.0},
        ])
        assert "x_1" in out and "x_2" in out
        assert "2" in out  # coefficient
        assert "= f" in out

    def test_zero_coeff_omitted(self):
        out = sturm_liouville.sl_formula_markdown([
            {"p": 0.0, "q": 1.0},
        ])
        assert "u" in out
        assert "partial" not in out


class TestBuildResultMat:
    def test_roundtrip_result_only(self):
        result = np.arange(12.0).reshape(3, 4)
        mat_bytes = sturm_liouville.build_result_mat(result)
        back = loadmat(io.BytesIO(mat_bytes))
        assert "result" in back
        np.testing.assert_array_almost_equal(back["result"], result)

    def test_roundtrip_with_error(self):
        result = np.ones((2, 2))
        error = np.zeros((2, 2))
        mat_bytes = sturm_liouville.build_result_mat(result, error=error)
        back = loadmat(io.BytesIO(mat_bytes))
        assert "result" in back and "error" in back
        np.testing.assert_array_almost_equal(back["error"], error)


class TestPlotResultAndError:
    def test_returns_figure_2d(self):
        arr = np.random.randn(10, 10)
        fig = sturm_liouville.plot_result_and_error(arr)
        assert fig is not None
        assert hasattr(fig, "axes")
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_returns_figure_with_error(self):
        arr = np.random.randn(8, 8)
        err = np.abs(arr)
        fig = sturm_liouville.plot_result_and_error(arr, error=err)
        assert len(fig.axes) >= 2
        import matplotlib.pyplot as plt
        plt.close(fig)


@pytest.mark.skipif(
    os.environ.get("CI", "").lower() not in ("0", "false", ""),
    reason="skip solver tests in CI when simulation.so may be missing",
)
class TestRunSturmLiouville:
    def test_apply_L_2d_small(self):
        try:
            from core import simulation_loader
            simulation_loader.get_simulation_module()
        except (ImportError, FileNotFoundError):
            pytest.skip("simulation.so not available")
        # 2D periodic, L = -Laplacian, apply L to u
        axes_config = [
            {"n_points": 8, "length": 2 * np.pi, "bc_from": "periodic", "bc_to": "periodic", "p": 1.0, "q": 0.0},
            {"n_points": 8, "length": 2 * np.pi, "bc_from": "periodic", "bc_to": "periodic", "p": 1.0, "q": 0.0},
        ]
        x = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        y = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="ij")
        u = np.sin(X) * np.cos(Y)
        mat_data = {"u": u}
        res = sturm_liouville.run_sturm_liouville(axes_config, mat_data)
        assert "result" in res
        assert res["result"].shape == (8, 8)
        # L = -nabla^2, L(sin x cos y) = (1+1) sin x cos y = 2 sin x cos y
        expected = 2.0 * u
        np.testing.assert_allclose(np.real(res["result"]), expected, atol=1e-10)

    def test_solve_L_2d_small(self):
        try:
            from core import simulation_loader
            simulation_loader.get_simulation_module()
        except (ImportError, FileNotFoundError):
            pytest.skip("simulation.so not available")
        axes_config = [
            {"n_points": 6, "length": 1.0, "bc_from": "dirichlet", "bc_to": "dirichlet", "p": 1.0, "q": 0.0},
            {"n_points": 6, "length": 1.0, "bc_from": "dirichlet", "bc_to": "dirichlet", "p": 1.0, "q": 0.0},
        ]
        f = np.ones((6, 6))
        mat_data = {"f": f}
        res = sturm_liouville.run_sturm_liouville(axes_config, mat_data)
        assert "result" in res
        assert res["result"].shape == (6, 6)

    def test_with_analytical_returns_error(self):
        try:
            from core import simulation_loader
            simulation_loader.get_simulation_module()
        except (ImportError, FileNotFoundError):
            pytest.skip("simulation.so not available")
        axes_config = [
            {"n_points": 4, "length": 1.0, "bc_from": "dirichlet", "bc_to": "dirichlet", "p": 1.0, "q": 0.0},
            {"n_points": 4, "length": 1.0, "bc_from": "dirichlet", "bc_to": "dirichlet", "p": 1.0, "q": 0.0},
        ]
        f = np.ones((4, 4))
        analytical = np.zeros((4, 4))
        mat_data = {"f": f, "analytical": analytical}
        res = sturm_liouville.run_sturm_liouville(axes_config, mat_data)
        assert "result" in res and "error" in res
        assert res["error"].shape == (4, 4)
