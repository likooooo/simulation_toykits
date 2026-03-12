"""Smoke tests for core.sturm_liouville."""

import io
import os

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from core import sturm_liouville

# -----------------------------------------------------------------------------
# Constants & helpers
# -----------------------------------------------------------------------------

MAT_FORMAT = "5"


def _make_mat_bytes(vars_dict):
    """Build .mat file bytes from a dict of variable names -> arrays."""
    buf = io.BytesIO()
    savemat(buf, vars_dict, format=MAT_FORMAT, do_compression=False)
    return buf.getvalue()


def _axes_2d_periodic(n, length=2 * np.pi):
    """2D periodic BC, L = -Laplacian (p=1, q=0)."""
    return [
        {"n_points": n, "length": length, "bc_from": "periodic", "bc_to": "periodic", "p": 1.0, "q": 0.0},
        {"n_points": n, "length": length, "bc_from": "periodic", "bc_to": "periodic", "p": 1.0, "q": 0.0},
    ]


def _axes_2d_dirichlet(n, length=1.0):
    """2D Dirichlet BC."""
    return [
        {"n_points": n, "length": length, "bc_from": "dirichlet", "bc_to": "dirichlet", "p": 1.0, "q": 0.0},
        {"n_points": n, "length": length, "bc_from": "dirichlet", "bc_to": "dirichlet", "p": 1.0, "q": 0.0},
    ]


def _skip_if_no_simulation():
    try:
        from core import simulation_loader
        simulation_loader.get_simulation_module()
    except (ImportError, FileNotFoundError):
        pytest.skip("simulation.so not available")


# -----------------------------------------------------------------------------
# LoadMatV7
# -----------------------------------------------------------------------------


class TestLoadMatV7:
    def test_load_from_bytes(self):
        u = np.zeros((8, 8))
        f = np.ones((8, 8))
        data = sturm_liouville.load_mat_v7(_make_mat_bytes({"u": u, "f": f}))
        assert "u" in data and "f" in data
        assert data["u"].shape == (8, 8)
        assert data["f"].shape == (8, 8)

    def test_load_rejects_dimension_mismatch(self):
        mat = _make_mat_bytes({
            "u": np.zeros((4, 4)),
            "f": np.zeros((3, 3)),
        })
        with pytest.raises(ValueError, match="维度必须一致"):
            sturm_liouville.load_mat_v7(mat)

    def test_load_requires_at_least_one_variable(self):
        mat = _make_mat_bytes({"x": np.zeros((2, 2))})
        with pytest.raises(ValueError, match="未找到变量"):
            sturm_liouville.load_mat_v7(mat)


# -----------------------------------------------------------------------------
# SlFormulaMarkdown
# -----------------------------------------------------------------------------


class TestSlFormulaMarkdown:
    def test_empty_axes(self):
        out = sturm_liouville.sl_formula_markdown([])
        assert "请添加坐标轴" in out

    def test_single_axis_has_f(self):
        out = sturm_liouville.sl_formula_markdown([{"p": 1.0, "q": 0.0}], has_f=True)
        assert "partial" in out and "x_1" in out
        assert "= f $$" in out

    def test_single_axis_no_f(self):
        out = sturm_liouville.sl_formula_markdown([{"p": 1.0, "q": 0.0}], has_f=False)
        assert "Lu =" in out and "partial" in out
        assert "= f" not in out

    def test_two_axes_with_coefficients(self):
        out = sturm_liouville.sl_formula_markdown([
            {"p": 1.0, "q": 0.0},
            {"p": 2.0, "q": 1.0},
        ], has_f=True)
        assert "x_1" in out and "x_2" in out
        assert "2" in out
        assert "= f $$" in out

    def test_zero_coeff_omitted(self):
        out = sturm_liouville.sl_formula_markdown([{"p": 0.0, "q": 1.0}])
        assert "u" in out
        assert "partial" not in out


# -----------------------------------------------------------------------------
# BuildResultMat
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# PlotResultAndError
# -----------------------------------------------------------------------------


class TestPlotResultAndError:
    def test_returns_figure_2d(self):
        arr = np.random.randn(10, 10)
        fig = sturm_liouville.plot_result_and_error(arr)
        assert fig is not None and hasattr(fig, "axes")
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_returns_figure_with_error(self):
        arr = np.random.randn(8, 8)
        fig = sturm_liouville.plot_result_and_error(arr, error=np.abs(arr))
        assert len(fig.axes) >= 2
        import matplotlib.pyplot as plt
        plt.close(fig)


# -----------------------------------------------------------------------------
# RunSturmLiouville (requires simulation.so)
# -----------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("CI", "").lower() not in ("0", "false", ""),
    reason="skip solver tests in CI when simulation.so may be missing",
)
class TestRunSturmLiouville:
    def test_apply_L_2d_periodic(self):
        _skip_if_no_simulation()
        n = 8
        axes_config = _axes_2d_periodic(n, length=1)
        x = np.linspace(0, axes_config[0]["length"], n, endpoint=False)
        y = np.linspace(0, axes_config[1]["length"], n, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="ij")
        # 需要整周期数据, 不然会出现频谱混叠
        u = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        res = sturm_liouville.run_sturm_liouville(axes_config, {"u": u})
        assert "result" in res and "input_field" in res
        assert res["result"].shape == (n, n)
        expected = -2.0 * (2 * np.pi)**2 * u
        np.testing.assert_allclose(np.real(res["result"]), expected, atol=1e-10)

    def test_solve_L_2d_dirichlet(self):
        _skip_if_no_simulation()
        axes_config = _axes_2d_dirichlet(6)
        f = np.ones((6, 6))
        res = sturm_liouville.run_sturm_liouville(axes_config, {"f": f})
        assert "result" in res
        assert res["result"].shape == (6, 6)

    def test_with_analytical_returns_error(self):
        _skip_if_no_simulation()
        axes_config = _axes_2d_dirichlet(4)
        f = np.ones((4, 4))
        analytical = np.zeros((4, 4))
        res = sturm_liouville.run_sturm_liouville(axes_config, {"f": f, "analytical": analytical})
        assert "result" in res and "error" in res
        assert res["error"].shape == (4, 4)
