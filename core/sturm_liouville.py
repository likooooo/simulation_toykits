"""Sturm-Liouville PDE calculator: stateless core (load mat, formula, solve, plot, export)."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from scipy.io import loadmat, savemat

from core import simulation_loader


def load_mat_v7(file_path_or_bytes: str | bytes) -> dict[str, Any]:
    """Load MATLAB file (v7 only). Returns dict with optional 'u', 'f', 'analytical' arrays.
    Validates that all present arrays have the same shape.
    Raises on v7.3 or dimension mismatch.
    """
    if isinstance(file_path_or_bytes, bytes):
        fp = io.BytesIO(file_path_or_bytes)
    else:
        fp = file_path_or_bytes

    try:
        raw = loadmat(fp, struct_as_record=False, squeeze_me=True)
    except Exception as e:
        msg = str(e).lower()
        if "v7.3" in msg or "hdf5" in msg or "h5" in msg:
            raise ValueError("仅支持 MATLAB v7 格式，不支持 v7.3 (HDF5)。请用 save(..., '-v7') 保存。") from e
        raise ValueError(f"无法读取 MATLAB 文件: {e}") from e

    # Strip internal names
    out = {}
    for key in ("u", "f", "analytical"):
        if key in raw and not key.startswith("__"):
            arr = raw[key]
            if hasattr(arr, "shape"):
                out[key] = np.atleast_1d(np.asarray(arr))
            else:
                out[key] = np.atleast_1d(np.asarray(arr))

    if not out:
        raise ValueError("MAT 文件中未找到变量 u、f 或 analytical。")

    shapes = [out[k].shape for k in out]
    if len(shapes) > 1 and len(set(shapes)) != 1:
        raise ValueError(
            f"变量 u、f、analytical 维度必须一致，当前: {dict((k, out[k].shape) for k in out)}"
        )
    return out


def sl_formula_markdown(axes_config: list[dict], has_f: bool = True) -> str:
    """Build markdown formula: Lu = {body} [= f if has_f].
    Rules: 0 -> omit term; 1 -> show term without coefficient; -1 -> show minus sign.
    Expands sum into one long formula (one term per axis).
    """
    if not axes_config:
        return "请添加坐标轴后刷新公式。"

    parts = []
    for i, ax in enumerate(axes_config):
        p = ax.get("p", 0.0)
        q = ax.get("q", 0.0)
        try:
            p = float(p)
            q = float(q)
        except (TypeError, ValueError):
            p = q = 0.0
        xi = f"x_{i+1}"
        term_parts = []
        if p != 0:
            if p == 1:
                term_parts.append(rf"+ \frac{{\partial^2 u}}{{\partial {xi}^2}}")
            elif p == -1:
                term_parts.append(rf"- \frac{{\partial^2 u}}{{\partial {xi}^2}}")
            else:
                term_parts.append(rf" {['-','+'][int(p>0)]} {np.abs(p)} \frac{{\partial^2 u}}{{\partial {xi}^2}}")
        if q != 0:
            if q == 1:
                term_parts.append(r"+ u")
            elif q == -1:
                term_parts.append(r"- u")
            else:
                term_parts.append(rf"{['-','+'][int(q>0)]} {np.abs(q)} u")
        if term_parts:
            parts.append(" ".join(term_parts))
    if not parts:
        formula_body = "所有系数为 0，无公式。"
    else:
        body = " ".join(parts)
        if body.startswith("+"):
            body = body[1:].lstrip()
        formula_body = f"$$ Lu = {body} = f $$" if has_f else f"$$ Lu = {body} $$"

    # Boundary conditions
    bc_lines = []
    for i, ax in enumerate(axes_config):
        bcf = ax.get("bc_from", "dirichlet")
        bct = ax.get("bc_to", "dirichlet")
        xi = f"x_{i+1}"
        if bcf == "periodic" or bct == "periodic":
            bc_lines.append(f"- **{xi}** (Periodic): \\( u(0) = u(L) \\)")
        elif bcf == "dirichlet" and bct == "dirichlet":
            bc_lines.append(f"- **{xi}** (Dirichlet): \\( u(0) = u(L) = 0 \\)")
        elif bcf == "neumann" and bct == "neumann":
            bc_lines.append(f"- **{xi}** (Neumann): \\( \\frac{{\\partial u}}{{\\partial n}} = 0 \\) at boundaries")
        else:
            bc_lines.append(f"- **{xi}**: from \\({bcf}\\), to \\({bct}\\)")
    bc_section = "\n\n**Boundary conditions**\n\n" + "\n".join(bc_lines) if bc_lines else ""
    return formula_body + bc_section


def _bc_to_enum(sim, bc: str) -> Any:
    m = {"periodic": sim.cartesian_boundary_condition.periodic,
         "dirichlet": sim.cartesian_boundary_condition.dirichlet,
         "neumann": sim.cartesian_boundary_condition.neumann}
    if bc not in m:
        raise ValueError(f"未知边界条件: {bc}，应为 periodic / dirichlet / neumann")
    return m[bc]


def run_sturm_liouville(axes_config: list[dict], mat_data: dict[str, np.ndarray]) -> dict[str, Any]:
    """Run SL solver. axes_config: list of {n_points, length, bc_from, bc_to, p, q}.
    mat_data: from load_mat_v7. Returns {result: ndarray, error?: ndarray}.
    - If u exists and f does not: result = L(u) (apply_L).
    - If f exists (u optional): solve L(u)=f, result = u (solve_L).
    """
    sim = simulation_loader.get_simulation_module()
    SlAxisCoeffs = sim.sl_axis_coeffs
    Solver = sim.general_sturm_liouville_solver_z

    shape = [int(ax["n_points"]) for ax in axes_config]
    lengths = [float(ax["length"]) for ax in axes_config]
    bc_from = [_bc_to_enum(sim, ax["bc_from"]) for ax in axes_config]
    bc_to = [_bc_to_enum(sim, ax["bc_to"]) for ax in axes_config]
    coeffs = []
    for ax in axes_config:
        c = SlAxisCoeffs()
        c.p = -float(ax.get("p", 1.0))
        c.q = -float(ax.get("q", 0.0))
        coeffs.append(c)
    has_u = "u" in mat_data
    has_f = "f" in mat_data
    has_analytical = "analytical" in mat_data

    if not has_u and not has_f:
        raise ValueError("MAT 数据中需至少包含 u 或 f。")

    # Reference shape from any of u/f/analytical
    ref_shape = mat_data.get("u", mat_data.get("f", mat_data.get("analytical"))).shape
    if tuple(shape) != ref_shape:
        raise ValueError(
            f"坐标轴网格与数据维度不一致: 配置 shape={shape}, 数据 shape={ref_shape}"
        )

    solver = Solver()
    solver.set_shape(shape)
    solver.set_lengths(lengths)
    solver.set_bc(bc_from, bc_to)
    solver.set_coeffs(coeffs)

    assert(has_f != has_u)
    tag = ["f", "u"][has_u]
    f = [lambda s: s.solve_L(), lambda s: s.apply_L()][has_u]
    input_field = np.asfortranarray(mat_data[tag], dtype=np.complex128)
    result = np.array(input_field, copy = True)
    solver.set_data_from_ndarray(result)
    solver.init()
    f(solver)

    out = {"result": result, "input_field": input_field}
    if has_analytical:
        analytical = np.asarray(mat_data["analytical"], dtype=np.complex128)
        if analytical.shape != result.shape:
            raise ValueError(
                f"analytical 与结果维度不一致: {analytical.shape} vs {result.shape}"
            )
        out["error"] = np.abs(result - analytical)
    return out


def build_result_mat(result: np.ndarray, error: np.ndarray | None = None) -> bytes:
    """Build .mat content as bytes (result; optional error)."""
    buf = io.BytesIO()
    d = {"result": result}
    if error is not None:
        d["error"] = error
    savemat(buf, d, format="5", do_compression=False)
    return buf.getvalue()


def plot_result_and_error(
    result: np.ndarray,
    error: np.ndarray | None = None,
    slice_kwargs: dict | None = None,
    input_field: np.ndarray | None = None,
):
    """Plot in one row: Input field | Output field | Error (if any). English titles. Returns matplotlib Figure."""
    import matplotlib.pyplot as plt

    slice_kwargs = slice_kwargs or {}
    nd = result.ndim
    result_real = np.real(result) if np.iscomplexobj(result) else result
    input_real = None
    if input_field is not None:
        input_real = np.real(input_field) if np.iscomplexobj(input_field) else input_field

    n_cols = 1
    if input_field is not None:
        n_cols += 1
    if error is not None:
        n_cols += 1
    # 固定总宽度 10 英寸，避免界面抖动
    fig, axes_arr = plt.subplots(1, n_cols, figsize=(10, 4))
    if n_cols == 1:
        axes_arr = [axes_arr]
    col = 0

    def plot_2d(ax, arr, title):
        im = ax.imshow(arr.T, origin="lower", aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_title(title)

    def plot_3d(ax, arr, title, sl):
        im = ax.imshow(arr[sl].T, origin="lower", aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_title(title)

    def plot_1d(ax, arr, title):
        ax.plot(np.arange(len(arr)), np.real(arr) if np.iscomplexobj(arr) else arr)
        ax.set_title(title)

    if nd == 2:
        if input_real is not None:
            plot_2d(axes_arr[col], input_real, "Input field")
            col += 1
        plot_2d(axes_arr[col], result_real, "Output field")
        col += 1
        if error is not None:
            im = axes_arr[col].imshow(error.T, origin="lower", aspect="auto", cmap="hot")
            plt.colorbar(im, ax=axes_arr[col])
            axes_arr[col].set_title("|result - analytical|")
    elif nd >= 3:
        idx = slice_kwargs.get("index", result.shape[-1] // 2)
        sl = (slice(None),) * (nd - 1) + (idx,)
        if input_real is not None:
            plot_3d(axes_arr[col], input_real, f"Input field (slice {idx})", sl)
            col += 1
        plot_3d(axes_arr[col], result_real, f"Output field (slice {idx})", sl)
        col += 1
        if error is not None:
            im = axes_arr[col].imshow(error[sl].T, origin="lower", aspect="auto", cmap="hot")
            plt.colorbar(im, ax=axes_arr[col])
            axes_arr[col].set_title(f"|error| (slice {idx})")
    else:
        if input_real is not None:
            plot_1d(axes_arr[col], input_real, "Input field")
            col += 1
        plot_1d(axes_arr[col], result_real, "Output field")
        col += 1
        if error is not None:
            axes_arr[col].plot(np.arange(len(error)), error)
            axes_arr[col].set_title("|error|")
    plt.tight_layout()
    return fig
