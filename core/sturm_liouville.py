"""Sturm-Liouville PDE calculator: stateless core (load mat, formula, solve, plot, export)."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from scipy.io import loadmat, savemat

from core import simulation_loader


def load_mat_v7(file_path_or_bytes: str | bytes) -> dict[str, Any]:
    """Load MATLAB file (v7 only). Returns dict of all non-internal array variables (any keys).
    Used to populate cache; no requirement for u/f/ut/analytical. All arrays must share the same shape.
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

    out = {}
    for key in raw:
        if key.startswith("__"):
            continue
        val = raw[key]
        try:
            arr = np.atleast_1d(np.asarray(val))
            if hasattr(arr, "shape") and arr.size > 0:
                out[key] = arr
        except (TypeError, ValueError):
            continue

    if not out:
        raise ValueError("MAT 文件中未找到可用的数组变量。")

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
    - 坐标轴 axes_config[i] 对应数组维度 i，数据 shape 须与 [ax['n_points'] for ax in axes_config] 一致；
      NumPy 默认 C 序，最后一维（索引 -1）在内存中变化最快。不修改 mat_data 中的原始数组。
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
    input_field = np.asarray(mat_data[tag], dtype=np.complex128).copy()
    result = input_field.copy()
    solver.set_data_from_ndarray(result)
    ec = solver.init()
    if ec != sim.axis_solver_init_error.ok:
        raise RuntimeError(f"solver.init() failed: {ec}")
    f(solver)

    out = {"result": result, "input_field": input_field}
    if has_analytical:
        analytical = np.asarray(mat_data["analytical"], dtype=np.complex128)
        if analytical.shape != result.shape:
            raise ValueError(
                f"analytical 与结果维度不一致: {analytical.shape} vs {result.shape}"
            )
        out["error"] = np.abs(result - analytical)
        out["analytical"] = analytical
    return out


def run_time_dependent_sturm_liouville(
    axes_config: list[dict],
    u0: np.ndarray,
    ut0: np.ndarray | None,
    t_start: float,
    t_end: float,
    dt: float,
    time_derivative_order: int = 2,
) -> dict[str, Any]:
    """Time-dependent solver (order 1/2): init once, then loop solver.solve(t) for each t.
    u0/ut0 在函数内持有直至时间循环结束，避免 solver 不拥有内存导致悬垂引用。
    Returns {frames: list of ndarray (complex), t_vals: ndarray}.
    If ut0 is None, use zeros like u0.
    """
    if not isinstance(axes_config, (list, tuple)):
        axes_config = []
    sim = simulation_loader.get_simulation_module()
    SlAxisCoeffs = sim.sl_axis_coeffs
    Solver = sim.time_dependent_sturm_liouville_solver_z

    shape = [int(ax["n_points"]) for ax in axes_config]
    lengths = [float(ax["length"]) for ax in axes_config]
    bc_from = [_bc_to_enum(sim, ax["bc_from"]) for ax in axes_config]
    bc_to = [_bc_to_enum(sim, ax["bc_to"]) for ax in axes_config]
    coeffs = []
    for ax in axes_config:
        c = SlAxisCoeffs()
        # i^2 项会多一个 -1, 然后移到时间项又多一个 -1, 所以这里 p 和 q 都是正的
        c.p = float(ax.get("p", 1.0))
        c.q = float(ax.get("q", 0.0))
        coeffs.append(c)
    time_derivative_order = int(time_derivative_order)
    if time_derivative_order not in (1, 2):
        raise ValueError(f"time_derivative_order must be 1 or 2, got {time_derivative_order}")

    u0_f = np.asarray(u0, dtype=np.complex128).copy()
    if ut0 is None:
        ut0_f = np.zeros_like(u0_f)
    else:
        ut0_f = np.asarray(ut0, dtype=np.complex128).copy()
    if u0_f.shape != ut0_f.shape:
        raise ValueError(f"u0 and ut0 shape mismatch: {u0_f.shape} vs {ut0_f.shape}")
    if tuple(shape) != u0_f.shape:
        raise ValueError(f"axes shape {shape} != u0 shape {u0_f.shape}")

    solver = Solver()
    solver.set_shape(shape)
    solver.set_lengths(lengths)
    solver.set_time_derivative_order(time_derivative_order)
    solver.set_bc(bc_from, bc_to)
    solver.set_coeffs(coeffs)
    ec = solver.init(u0_f, ut0_f)
    if ec != sim.axis_solver_init_error.ok:
        raise RuntimeError(f"time_dependent solver.init() failed: {ec}")
    u0_shape = u0_f.shape
    result = np.empty(u0_shape, dtype=np.complex128)
    frames = []
    t_vals = []
    t = t_start
    while t <= t_end + 1e-12 * abs(dt):
        solver.solve(t, result)
        frames.append(np.array(result, copy=True))
        t_vals.append(t)
        t += dt
    return {"frames": frames, "t_vals": np.array(t_vals)}


def _to_real_array(data: np.ndarray, mode: str = "real") -> np.ndarray:
    """Scalar field from frame; mode in 'real'/'imag'/'amplitude'/'phase' (phase: angle in [-π,π] → [-1,1])."""
    data = np.asarray(data)
    if np.iscomplexobj(data):
        if mode == "phase":
            data = np.angle(data) / np.pi  # [-1, 1]
        else:
            data = {"real": data.real, "imag": data.imag, "amplitude": np.abs(data)}[mode]
    return np.asarray(data, dtype=float)


def _map_to_minus1_1_with_range(data: np.ndarray, lb: float, ub: float) -> np.ndarray:
    """Linearly map data to [-1, 1] using fixed lb, ub. Match visualizer.py."""
    data = np.asarray(data, dtype=float)
    if ub == lb:
        return np.zeros_like(data)
    out = (data - lb) / (ub - lb) * 2.0 - 1.0
    return np.clip(out, -1.0, 1.0)


def build_result_mat(result: np.ndarray, error: np.ndarray | None = None) -> bytes:
    """Build .mat content as bytes (result; optional error)."""
    buf = io.BytesIO()
    d = {"result": result}
    if error is not None:
        d["error"] = error
    savemat(buf, d, format="5", do_compression=False)
    return buf.getvalue()


def _slice_for_nd(arr: np.ndarray, slice_kwargs: dict) -> np.ndarray:
    """For ndim>=3, return xy-plane slice at middle z; else return arr."""
    nd = arr.ndim
    if nd <= 2:
        return arr
    idx = slice_kwargs.get("index", arr.shape[-1] // 2)
    sl = (slice(None),) * (nd - 1) + (idx,)
    return arr[sl]


def plot_result_and_error(
    result: np.ndarray,
    error: np.ndarray | None = None,
    slice_kwargs: dict | None = None,
    input_field: np.ndarray | None = None,
    analytical: np.ndarray | None = None,
):
    """Plot: Input (amplitude + phase) | Output (amplitude + phase) | Error (amplitude + phase, if any).
    Order top to bottom: input figure, output figure, error figure with two subplots (optional).
    Error row: amplitude error | |result| - |analytical|, phase error (wrapped angle difference).
    1D: curves; 2D: images; 3D+: middle z-slice as 2D. Returns single matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    slice_kwargs = slice_kwargs or {}
    nd = result.ndim
    # For 3D+ use middle slice for display
    result_2d = _slice_for_nd(result, slice_kwargs)
    input_2d = _slice_for_nd(input_field, slice_kwargs) if input_field is not None else None
    analytical_2d = _slice_for_nd(analytical, slice_kwargs) if analytical is not None else None
    error_2d = _slice_for_nd(error, slice_kwargs) if error is not None else None
    display_ndim = result_2d.ndim

    n_rows = 1
    if input_field is not None:
        n_rows += 1
    if error is not None:
        n_rows += 1
    fig = plt.figure(figsize=(10, 4 * n_rows))
    gs = fig.add_gridspec(n_rows, 2)

    def plot_amp_phase(ax_amp, ax_phase, arr, title_prefix):
        if np.iscomplexobj(arr):
            amp, phase = np.abs(arr), np.angle(arr)
        else:
            amp, phase = arr, np.zeros_like(arr)
        if display_ndim == 1:
            x = np.arange(len(amp))
            ax_amp.plot(x, amp)
            ax_amp.set_title(f"{title_prefix} — Amplitude")
            ax_phase.plot(x, phase)
            ax_phase.set_title(f"{title_prefix} — Phase")
        else:
            im0 = ax_amp.imshow(amp.T, origin="lower", aspect="auto")
            plt.colorbar(im0, ax=ax_amp)
            ax_amp.set_title(f"{title_prefix} — Amplitude")
            im1 = ax_phase.imshow(phase.T, origin="lower", aspect="auto", cmap="twilight")
            plt.colorbar(im1, ax=ax_phase)
            ax_phase.set_title(f"{title_prefix} — Phase")

    def plot_error_amp_phase(ax_amp_err, ax_phase_err, res, ana):
        """Plot amplitude error and phase error (two subplots). res, ana complex arrays."""
        amp_err = np.abs(np.abs(res) - np.abs(ana))
        phase_diff = np.angle(res) - np.angle(ana)
        phase_err = np.abs(np.angle(np.exp(1j * phase_diff)))  # wrapped to [0, pi]
        if display_ndim == 1:
            x = np.arange(len(amp_err))
            ax_amp_err.plot(x, amp_err)
            ax_amp_err.set_title("Error — Amplitude")
            ax_phase_err.plot(x, phase_err)
            ax_phase_err.set_title("Error — Phase")
        else:
            im0 = ax_amp_err.imshow(amp_err.T, origin="lower", aspect="auto", cmap="hot")
            plt.colorbar(im0, ax=ax_amp_err)
            ax_amp_err.set_title("Error — Amplitude")
            im1 = ax_phase_err.imshow(phase_err.T, origin="lower", aspect="auto", cmap="hot")
            plt.colorbar(im1, ax=ax_phase_err)
            ax_phase_err.set_title("Error — Phase")

    row = 0
    if input_2d is not None:
        plot_amp_phase(fig.add_subplot(gs[row, 0]), fig.add_subplot(gs[row, 1]), input_2d, "Input")
        row += 1
    plot_amp_phase(fig.add_subplot(gs[row, 0]), fig.add_subplot(gs[row, 1]), result_2d, "Output")
    row += 1
    if error_2d is not None:
        if analytical_2d is not None:
            plot_error_amp_phase(
                fig.add_subplot(gs[row, 0]), fig.add_subplot(gs[row, 1]), result_2d, analytical_2d
            )
        else:
            # fallback: single combined error (legacy)
            ax_err = fig.add_subplot(gs[row, :])
            if error_2d.ndim == 1:
                ax_err.plot(np.arange(len(error_2d)), error_2d)
            else:
                im = ax_err.imshow(error_2d.T, origin="lower", aspect="auto", cmap="hot")
                plt.colorbar(im, ax=ax_err)
            ax_err.set_title("|Error|")
    plt.tight_layout()
    return fig


def plot_error_only(
    result: np.ndarray,
    analytical: np.ndarray,
    slice_kwargs: dict | None = None,
):
    """Plot error as two subplots: amplitude error and phase error. Returns matplotlib Figure."""
    import matplotlib.pyplot as plt

    slice_kwargs = slice_kwargs or {}
    result_2d = _slice_for_nd(result, slice_kwargs)
    analytical_2d = _slice_for_nd(analytical, slice_kwargs)
    display_ndim = result_2d.ndim

    fig, (ax_amp, ax_phase) = plt.subplots(1, 2, figsize=(10, 4))
    amp_err = np.abs(np.abs(result_2d) - np.abs(analytical_2d))
    phase_diff = np.angle(result_2d) - np.angle(analytical_2d)
    phase_err = np.abs(np.angle(np.exp(1j * phase_diff)))
    if display_ndim == 1:
        x = np.arange(len(amp_err))
        ax_amp.plot(x, amp_err)
        ax_phase.plot(x, phase_err)
    else:
        im0 = ax_amp.imshow(amp_err.T, origin="lower", aspect="auto", cmap="hot")
        plt.colorbar(im0, ax=ax_amp)
        im1 = ax_phase.imshow(phase_err.T, origin="lower", aspect="auto", cmap="hot")
        plt.colorbar(im1, ax=ax_phase)
    ax_amp.set_title("Error — Amplitude")
    ax_phase.set_title("Error — Phase")
    plt.tight_layout()
    return fig

    return fig
