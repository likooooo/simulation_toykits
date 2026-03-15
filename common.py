"""Fresnel 页面共用的 session 初始化与对 core 的薄封装。"""

import base64
import io
from pathlib import Path
from typing import Any, Callable, List

import pandas as pd
import streamlit as st


# 固定尺寸图：占页面 80% 宽（通过 CSS 实现，避免 use_column_width 弃用与抖动）
PYPLOT_PAGE_WIDTH_RATIO = 0.8


def pyplot_fixed_width(fig, width: int = None, dpi: int = 100):
    """将 matplotlib Figure 以 80% 页宽渲染（CSS），避免界面抖动；传 width 时按像素固定宽。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    if width is not None:
        st.image(buf, width=width)
    else:
        b64 = base64.b64encode(buf.getvalue()).decode()
        ratio_pct = int(PYPLOT_PAGE_WIDTH_RATIO * 100)
        st.markdown(
            f'<div style="width:{ratio_pct}%; margin:0 auto;">'
            f'<img src="data:image/png;base64,{b64}" style="width:100%; height:auto; display:block;"/>'
            "</div>",
            unsafe_allow_html=True,
        )


def video_fixed_width(video_bytes: bytes, format: str = "video/mp4"):
    """与 pyplot_fixed_width 一致：视频占页宽 80%（通过 st.columns 实现）。"""
    ratio = PYPLOT_PAGE_WIDTH_RATIO
    left = (1 - ratio) / 2
    right = (1 - ratio) / 2
    cols = st.columns([left, ratio, right])
    with cols[1]:
        st.video(video_bytes, format=format)


# 全 pages 统一风格的「保存结果」下载按钮（.mat / .mp4 等）
def save_result_mat_button(data: bytes, file_name: str, key: str, label: str | None = None, mime: str | None = None):
    """渲染统一风格的保存结果下载按钮。未传 label/mime 时按 file_name 后缀推断。"""
    if label is None:
        suffix = (file_name or "").strip().split(".")[-1].lower()
        label = "💾 保存结果 (.mat)" if suffix == "mat" else (f"💾 保存结果 ({suffix})" if suffix else "💾 保存结果")
    if mime is None:
        suffix = (file_name or "").strip().split(".")[-1].lower()
        mime = "video/mp4" if suffix == "mp4" else "application/octet-stream"
    return st.download_button(
        label=label,
        data=data,
        file_name=file_name,
        mime=mime,
        key=key,
        width="stretch",
    )


def build_beam_result_mat(field: Any, meta: dict) -> bytes:
    """由 beam 页的 field 与 meta 生成 .mat 字节。"""
    from scipy.io import savemat

    buf = io.BytesIO()
    savemat(buf, {"field": field}, format="5", do_compression=False)
    return buf.getvalue()


def build_film_result_mat(result: Any, wl: float, angle_deg: float) -> bytes:
    """由 films 页的 Fresnel 结果生成 .mat 字节（R/T、r/t、波长、角度）。"""
    from scipy.io import savemat

    buf = io.BytesIO()
    savemat(
        buf,
        {
            "R_s": result.R_s,
            "T_s": result.T_s,
            "R_p": result.R_p,
            "T_p": result.T_p,
            "r_s": result.r_s,
            "t_s": result.t_s,
            "r_p": result.r_p,
            "t_p": result.t_p
        },
        format="5",
        do_compression=False,
    )
    return buf.getvalue()


def build_spectral_curve_result_mat(data: dict) -> bytes:
    """由光谱曲线页的结果数据生成 .mat 字节（不含 meta）。"""
    from scipy.io import savemat

    buf = io.BytesIO()
    savemat(buf, data, format="5", do_compression=False)
    return buf.getvalue()


def build_angular_map_result_mat(data: dict) -> bytes:
    """由角度-光谱图页的结果数据生成 .mat 字节（不含 meta）。"""
    from scipy.io import savemat

    buf = io.BytesIO()
    savemat(buf, data, format="5", do_compression=False)
    return buf.getvalue()

from core import materials as core_materials
from core import refractiveindex as ri


def init_materials_db():
    st.session_state["materials_db"] = {}
    st.session_state["materials_db"]["Vacuum"] = {
        "Shelf ID": "\\",
        "Book ID": "Vacuum",
        "Page ID": "\\",
        "Material Name": "Vacuum",
        "Data Source": "\\",
    }


def init_layer_config():
    st.session_state["layer_config"] = pd.DataFrame(
        [{"Material": "Vacuum", "Thickness (um)": 0.0, "n": 1, "k": 0}]
    )


def ensure_fresnel_session_state():
    """保证 Fresnel 相关页面用到的 session 键都存在（新 session 或直接打开子页时必调）。"""
    if "materials_db" not in st.session_state:
        init_materials_db()
    if "layer_config" not in st.session_state:
        init_layer_config()
    if "wavelength" not in st.session_state:
        st.session_state["wavelength"] = 0.532
    if "degree" not in st.session_state:
        st.session_state["degree"] = 15
    if "film_stack_code" not in st.session_state:
        st.session_state["film_stack_code"] = (
            "Vacuum 0 1 0 SiO2  0.12874 1.4621 1.4254e-5 Ta2O5  0.04396 2.1548  0.00021691 "
            "SiO2 0.27602 1.4621 1.4254e-5 Ta2O5 0.01699 2.1548  0.00021691  "
            "SiO2  0.24735 1.4621 1.4254e-5 fused_silica 0 1.4607 0"
        )


ensure_fresnel_session_state()


def get_nk_at_wavelength(name, wl_um):
    """从 session 材料库与波长取 nk，依赖 core.materials；错误时展示 st.error 并返回 1+0j。"""
    materials_db = st.session_state.get("materials_db", {})
    try:
        return core_materials.get_nk_at_wavelength(
            materials_db, name, wl_um, ri.load_nk
        )
    except Exception:
        st.error(
            f"加载材料 {name} (@ {wl_um} um) 出错.\n"
            "1. 请在 Material Database 中添加材料;\n2. 检查材料波长在范围内;\n"
        )
        return 1.0 + 0.0j


def with_nk_columns(df, wl_um):
    """为层配置表补全 n、k 列，通过 core.materials；get_nk 使用当前 session 材料库。"""
    materials_db = st.session_state.get("materials_db", {})

    def get_nk(name):
        try:
            return core_materials.get_nk_at_wavelength(
                materials_db, name, wl_um, ri.load_nk
            )
        except Exception:
            return 1.0 + 0.0j

    return core_materials.with_nk_columns(df, wl_um, get_nk)


def ensure_beams_session_state():
    """Ensure beams_result_cache exists so each beam page can store/restore results."""
    if "beams_result_cache" not in st.session_state:
        st.session_state["beams_result_cache"] = {}


def page_key_from_file(file_path: str) -> str:
    """Stable page key from script path, e.g. 'pages/beams_caculator/plane-wave.py' -> 'plane-wave'."""
    return Path(file_path).stem


# PDE 计算器：缓存 key -> 用户展示名称（beams 页 stem -> 显示名；fresnel 子项用 data key）
PDE_CACHE_KEY_TO_LABEL = {
    "flat top beam": "Flat-Top Beam",
    "plane wave": "Plane Wave",
    "spherical wave": "Spherical Wave",
    "quadratic wave": "Quadratic Wave",
    "hermite gaussian beam": "Hermite-Gaussian Beam",
    "laguerre gaussian beam": "Laguerre-Gaussian Beam",
    "spectral_R_s": "光谱曲线 R_s (TE 反射)",
    "spectral_R_p": "光谱曲线 R_p (TM 反射)",
    "spectral_T_s": "光谱曲线 T_s (TE 透射)",
    "spectral_T_p": "光谱曲线 T_p (TM 透射)",
    "angular_R_s": "角度-光谱图 R_s (TE)",
    "angular_T_s": "角度-光谱图 T_s (TE)",
    "angular_R_p": "角度-光谱图 R_p (TM)",
    "angular_T_p": "角度-光谱图 T_p (TM)",
}


def get_pde_cache_options(session_state) -> List[dict]:
    """Collect 1D/2D arrays from beams_result_cache and fresnel caches for PDE input.
    Returns list of {"id": str, "label": str, "array": np.ndarray} (id unique for dropdown value).
    """
    import numpy as np

    out = []
    # Beams: each page cache has "field" (2D complex)
    beams = session_state.get("beams_result_cache") or {}
    for page_key, entry in beams.items():
        if not isinstance(entry, dict) or "field" not in entry:
            continue
        arr = entry["field"]
        if not hasattr(arr, "shape") or arr.ndim < 1 or arr.ndim > 2:
            continue
        label = PDE_CACHE_KEY_TO_LABEL.get(page_key, f"{page_key} (场)")
        out.append({"id": f"beams__{page_key}__field", "label": f"Beams / {label}", "array": np.asarray(arr)})

    # Fresnel spectral: 1D arrays R_s, R_p, T_s, T_p
    spectral = session_state.get("spectral_result_data") or {}
    for key in ("R_s", "R_p", "T_s", "T_p"):
        if key not in spectral:
            continue
        arr = np.asarray(spectral[key])
        if arr.ndim != 1:
            continue
        label = PDE_CACHE_KEY_TO_LABEL.get(f"spectral_{key}", f"光谱曲线 {key}")
        out.append({"id": f"fresnel__spectral__{key}", "label": f"Fresnel / {label}", "array": arr})

    # Fresnel angular: 2D arrays
    angular = session_state.get("angular_result_data") or {}
    for key in ("R_s", "T_s", "R_p", "T_p"):
        if key not in angular:
            continue
        arr = np.asarray(angular[key])
        if arr.ndim != 2:
            continue
        label = PDE_CACHE_KEY_TO_LABEL.get(f"angular_{key}", f"角度-光谱图 {key}")
        out.append({"id": f"fresnel__angular__{key}", "label": f"Fresnel / {label}", "array": arr})

    # PDE 计算器本身的上次输出结果（命名为 输入名 (result)）
    res = session_state.get("pde_sl_result")
    if res and "result" in res:
        arr = np.asarray(res["result"])
        if hasattr(arr, "shape") and arr.ndim >= 1:
            label = session_state.get("pde_sl_result_cache_label", "PDE 计算器 / 上次输出结果")
            out.append({"id": "pde__last_result", "label": label, "array": arr})

    # Time-dependent SL 上次输出（某一帧或首帧）
    tdsl = session_state.get("pde_tdsl_result")
    if tdsl and "frames" in tdsl and tdsl["frames"]:
        arr = np.asarray(tdsl["frames"][0])
        if hasattr(arr, "shape") and arr.ndim >= 1:
            out.append({"id": "pde_tdsl__last", "label": "PDE 时变波 / 上次结果首帧", "array": arr})

    # 导入的 .mat 数据（仅保留最近一次，命名如 u (文件名.mat) / analytical (文件名.mat) / u'(0) (文件名.mat)）
    mat_import = session_state.get("pde_mat_import_cache")
    if isinstance(mat_import, dict) and "entries" in mat_import:
        for ent in mat_import["entries"]:
            if isinstance(ent, dict) and "key" in ent and "label" in ent and "array" in ent:
                arr = np.asarray(ent["array"])
                if hasattr(arr, "shape") and arr.ndim >= 1:
                    out.append({"id": ent.get("id", f"pde_mat__{ent['key']}"), "label": ent["label"], "array": arr})

    return out


def page_grid_init(key_prefix=""):
    """Render grid inputs, return (x_min, x_max, y_min, y_max, nx, ny)."""
    p = f"{key_prefix}_" if key_prefix else ""
    st.caption("Grid parameters:")
    x_min = st.number_input("start x (µm)", value=-1.1, format="%.2f", key=f"{p}xmin")
    x_max = st.number_input("end   x (µm)", value=1.1, format="%.2f", key=f"{p}xmax")
    y_min = st.number_input("start y (µm)", value=-1.1, format="%.2f", key=f"{p}ymin")
    y_max = st.number_input("end   y (µm)", value=1.1, format="%.2f", key=f"{p}ymax")
    nx = st.number_input("nx", value=100, min_value=2, max_value=256, step=1, key=f"{p}nx")
    ny = st.number_input("ny", value=100, min_value=2, max_value=256, step=1, key=f"{p}ny")
    return x_min, x_max, y_min, y_max, nx, ny


def page_plane_wave_init(key_prefix="pw"):
    """Render plane wave params, return (wavelength, theta_deg, phi_deg)."""
    st.caption("Beam parameters:")
    wavelength = st.number_input("Wavelength (µm)", value=0.11, format="%.4f", key=f"{key_prefix}_wl")
    theta_deg = st.number_input("θ (deg)", min_value = -89.0, max_value = 89.0, value = 10.0, step = 0.1, key=f"{key_prefix}_theta")
    phi_deg = st.number_input("φ (deg)", min_value = -180.0, max_value = 180.0, value = 30.0, step = 0.1, key=f"{key_prefix}_phi")
    return wavelength, theta_deg, phi_deg


def page_quadratic_wave_init(key_prefix="qw"):
    """Render quadratic wave params, return (wavelength, z_ratio)."""
    st.caption("Beam parameters:")
    wavelength = st.number_input("Wavelength (µm)", value=1.1, format="%.4f", key=f"{key_prefix}_wl")
    z_ratio = st.number_input("z ratio (z = z ratio × wavelength)", value=0.25, format="%.2f", key=f"{key_prefix}_zr")
    return wavelength, z_ratio


def page_spherical_wave_init(key_prefix="sw"):
    """Render spherical wave params, return (wavelength, z_ratio)."""
    st.caption("Beam parameters:")
    wavelength = st.number_input("Wavelength (µm)", value=1.1, format="%.4f", key=f"{key_prefix}_wl")
    z_ratio = st.number_input("z ratio (z = z ratio × wavelength)", value=0.25, format="%.2f", key=f"{key_prefix}_zr")
    return wavelength, z_ratio


def page_flat_top_init(key_prefix="ft"):
    """Render flat-top params; return (mode, fraction, r, order, rx, ry, order_x, order_y)."""
    st.caption("Beam parameters:")
    mode = st.radio("Mode", ["Circular", "Rectangular"], key=f"{key_prefix}_mode", horizontal=True)
    fraction = st.number_input(
        "amplitude at radius", value=0.5, min_value=1e-6, max_value=1 - 1e-6, format="%.2f", key=f"{key_prefix}_frac"
    )
    if mode == "Circular":
        r = st.number_input("r (µm)", value=0.8, min_value=0.01, format="%.2f", key=f"{key_prefix}_r")
        order = st.number_input("order", value=5.5, format="%.2f", key=f"{key_prefix}_order")
        rx = ry = order_x = order_y = None
    else:
        rx = st.number_input("rx (µm)", value=0.8, min_value=0.01, format="%.2f", key=f"{key_prefix}_rx")
        ry = st.number_input("ry (µm)", value=0.8, min_value=0.01, format="%.2f", key=f"{key_prefix}_ry")
        order_x = st.number_input("order x", value=5.5, format="%.2f", key=f"{key_prefix}_ox")
        order_y = st.number_input("order y", value=5.5, format="%.2f", key=f"{key_prefix}_oy")
        r = order = None
    return mode, fraction, r, order, rx, ry, order_x, order_y


def page_hermite_gaussian_init(key_prefix="hg"):
    """Render Hermite-Gaussian params, return (wavelength, m, n, z, wx0, wy0)."""
    st.caption("Beam parameters:")
    wavelength = st.number_input("Wavelength (µm)", value=0.5, format="%.4f", key=f"{key_prefix}_wl")
    z_ratio = st.number_input("z ratio (z = z ratio × wavelength)", value=0.25, format="%.2f", key=f"{key_prefix}_zr")
    wx0 = st.number_input("wx0 (µm)", value=1.0, min_value=0.01, format="%.2f", key=f"{key_prefix}_wx0")
    wy0 = st.number_input("wy0 (µm)", value=1.0, min_value=0.01, format="%.2f", key=f"{key_prefix}_wy0")
    m = st.number_input("m", value=3, min_value=0, step=1, key=f"{key_prefix}_m")
    n = st.number_input("n", value=3, min_value=0, step=1, key=f"{key_prefix}_n")
    return wavelength, m, n, z_ratio*wavelength, wx0, wy0


def page_laguerre_gaussian_init(key_prefix="lg"):
    """Render Laguerre-Gaussian params, return (wavelength, p, l, z, w0)."""
    st.caption("Beam parameters:")
    wavelength = st.number_input("Wavelength (µm)", value=0.5, format="%.4f", key=f"{key_prefix}_wl")
    z_ratio = st.number_input("z ratio (z = z ratio × wavelength)", value=0.25, format="%.2f", key=f"{key_prefix}_zr")
    w0 = st.number_input("w0 (µm)", value=1.0, min_value=0.01, format="%.2f", key=f"{key_prefix}_w0")
    p = st.number_input("p", value=3, min_value=0, step=1, key=f"{key_prefix}_p")
    l = st.number_input("l", value=-3, step=1, key=f"{key_prefix}_l")
    return wavelength, p, l, z_ratio*wavelength, w0


def render_table_editor(
    key_prefix: str,
    columns: List[dict],
    items: List[Any],
    render_row: Callable[[int, Any, List], None],
    on_add: Callable[[], None],
    on_clear: Callable[[], None],
    on_delete: Callable[[int], None],
    add_label: str = "➕ 添加",
    clear_label: str = "🗑️ 清空",
    delete_label: str = "删除",
    left_buttons: List[dict] = None,
) -> None:
    """
    统一表格式编辑 UI：标题行仅第一行显示，每行最后一列为删除按钮，标题前一行有添加/清空按钮。
    - left_buttons: 可选，[{"label": str, "key": str, "on_click": callable}, ...]，显示在添加按钮左侧。
    - columns: [{"label": "列名", "width": 1}, ...]，width 为列宽比例。
    - items: 当前行数据列表（如 list of dict）。
    - render_row(row_index, item, cols): 在 cols[0], cols[1], ... 中渲染该行控件（不含删除列）。
    - on_add / on_clear / on_delete(i): 添加、清空、删除第 i 行的回调（可在内部 st.rerun()）。
    """
    widths = [c["width"] for c in columns]
    op_width = 0.6
    left_buttons = left_buttons or []
    # 标题前一行：左侧按钮（如刷新坐标轴）、添加、清空
    n_left = len(left_buttons)
    n_total = n_left + 2
    row_cols = st.columns([1] * n_left + [1, 1])
    for i, lb in enumerate(left_buttons):
        with row_cols[i]:
            if st.button(lb["label"], key=lb["key"], width="stretch"):
                lb["on_click"]()
    with row_cols[n_left]:
        if st.button(add_label, key=f"{key_prefix}_add", width="stretch"):
            on_add()
    with row_cols[n_left + 1]:
        if st.button(clear_label, key=f"{key_prefix}_clear", width="stretch"):
            on_clear()
    # 唯一一行标题
    header_cols = st.columns(widths + [op_width])
    for j, col_def in enumerate(columns):
        with header_cols[j]:
            st.markdown(f"**{col_def['label']}**")
    with header_cols[-1]:
        st.markdown("**操作**")
    # 数据行：仅内容 + 最后一列删除
    for i, item in enumerate(items):
        row_cols = st.columns(widths + [op_width])
        render_row(i, item, row_cols[:-1])
        with row_cols[-1]:
            st.button(
                delete_label,
                key=f"{key_prefix}_del_{i}",
                on_click=on_delete,
                args=(i,),
            )


# --- 从 session 读取当前参数，供 @st.fragment 内计算使用，避免整页 rerun 导致控件抖动 ---
def _p(key_prefix, name, default):
    return st.session_state.get(f"{key_prefix}_{name}", default)


def get_grid_params_from_session(key_prefix):
    """Same order as page_grid_init return: (x_min, x_max, y_min, y_max, nx, ny)."""
    return (
        float(_p(key_prefix, "xmin", -1.1)),
        float(_p(key_prefix, "xmax", 1.1)),
        float(_p(key_prefix, "ymin", -1.1)),
        float(_p(key_prefix, "ymax", 1.1)),
        int(_p(key_prefix, "nx", 100)),
        int(_p(key_prefix, "ny", 100)),
    )


def get_plane_wave_params_from_session(key_prefix):
    return (
        float(_p(key_prefix, "wl", 0.11)),
        float(_p(key_prefix, "theta", 10.0)),
        float(_p(key_prefix, "phi", 30.0)),
    )


def get_quadratic_wave_params_from_session(key_prefix):
    return (float(_p(key_prefix, "wl", 1.1)), float(_p(key_prefix, "zr", 0.25)))


def get_spherical_wave_params_from_session(key_prefix):
    return (float(_p(key_prefix, "wl", 1.1)), float(_p(key_prefix, "zr", 0.25)))


def get_flat_top_params_from_session(key_prefix):
    mode = _p(key_prefix, "mode", "Circular")
    fraction = float(_p(key_prefix, "frac", 0.5))
    if mode == "Circular":
        return (
            mode,
            fraction,
            float(_p(key_prefix, "r", 0.8)),
            float(_p(key_prefix, "order", 5.5)),
            None,
            None,
            None,
            None,
        )
    return (
        mode,
        fraction,
        None,
        None,
        float(_p(key_prefix, "rx", 0.8)),
        float(_p(key_prefix, "ry", 0.8)),
        float(_p(key_prefix, "ox", 5.5)),
        float(_p(key_prefix, "oy", 5.5)),
    )


def get_hermite_gaussian_params_from_session(key_prefix):
    wl = float(_p(key_prefix, "wl", 0.5))
    zr = float(_p(key_prefix, "zr", 0.25))
    return (
        wl,
        int(_p(key_prefix, "m", 3)),
        int(_p(key_prefix, "n", 3)),
        zr * wl,
        float(_p(key_prefix, "wx0", 1.0)),
        float(_p(key_prefix, "wy0", 1.0)),
    )


def get_laguerre_gaussian_params_from_session(key_prefix):
    wl = float(_p(key_prefix, "wl", 0.5))
    zr = float(_p(key_prefix, "zr", 0.25))
    return (
        wl,
        int(_p(key_prefix, "p", 3)),
        int(_p(key_prefix, "l", -3)),
        zr * wl,
        float(_p(key_prefix, "w0", 1.0)),
    )
