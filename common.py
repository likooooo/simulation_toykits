"""Shared session helpers and thin wrappers for Gaussian optics / workspace pages."""

import io
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import streamlit as st

from simulation_database.workspace import ensure_sim_workspace, get_workspace_materials


from filmstack_simulation.simulation import nk_at_wavelength as _material_nk_at_wavelength

HOST_DESIGN_TOKENS_PATH = Path(__file__).resolve().parent / "ui" / "design_tokens.css"


def show_markdown_file(file_path: str | Path) -> None:
    path = Path(file_path)
    if path.is_file():
        content = path.read_text(encoding="utf-8")
        st.markdown(content, unsafe_allow_html=True)
    else:
        st.error(f"找不到文件: {file_path}")


def _lookup_nk_at_wavelength(materials_db: Dict[str, Any], name: str, wl_um: float) -> complex:
    """根据材料库与波长返回复折射率 n + 1j*k。"""
    if name == "Vacuum":
        return 1.0 + 0.0j
    mat = materials_db.get(name)
    if mat is None:
        return 1.0 + 0.0j
    return _material_nk_at_wavelength(mat, wl_um)


def pyplot_fixed_width(fig, width: int = None, dpi: int = 100):
    """将 matplotlib Figure 以 80% 页宽渲染（CSS），避免界面抖动；传 width 时按像素固定宽。"""
    if width is not None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        st.image(buf, width=width)
        return
    from filmstack_simulation.plots import show_figure

    show_figure(fig, dpi=dpi)


def video_fixed_width(video_bytes: bytes, format: str = "video/mp4"):
    """与 pyplot_fixed_width 一致：视频占页宽 80%（通过 st.columns 实现）。"""
    from filmstack_simulation.plots import PAGE_WIDTH_RATIO

    ratio = PAGE_WIDTH_RATIO
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


def ensure_fresnel_session_state():
    """Ensure session keys used by diffraction / workspace pages exist."""
    ensure_sim_workspace()
    if "wavelength" not in st.session_state:
        st.session_state["wavelength"] = 0.532


def get_nk_at_wavelength(name, wl_um):
    """Look up nk from workspace materials; show st.error and re-raise on failure."""
    materials_db = get_workspace_materials()
    try:
        return _lookup_nk_at_wavelength(materials_db, name, wl_um)
    except Exception as exc:
        st.error(
            f"加载材料 {name} (@ {wl_um} μm) 出错.\n"
            "1. 请在仿真数据库工作区中加入该材料;\n2. 检查材料波长在范围内;\n"
        )
        raise exc


def ensure_beams_session_state():
    """Ensure beams_result_cache exists so each beam page can store/restore results."""
    if "beams_result_cache" not in st.session_state:
        st.session_state["beams_result_cache"] = {}


def page_key_from_file(file_path: str) -> str:
    """Stable page key from script path, e.g. 'pages/gaussian_optics_toolkits/plane wave.py' -> 'plane wave'."""
    return Path(file_path).stem


# PDE 计算器：缓存 key -> 用户展示名称（beams 页 stem -> 显示名；fresnel 子项用 data key）
PDE_CACHE_KEY_TO_LABEL = {
    "flat top beam": "Flat-Top Beam",
    "plane wave": "Plane Wave",
    "spherical wave": "Spherical Wave",
    "quadratic wave": "Quadratic Wave",
    "hermite gaussian beam": "Hermite-Gaussian Beam",
    "laguerre gaussian beam": "Laguerre-Gaussian Beam",
}


def get_available_materials() -> List[str]:
    """Material names from workspace (deduplicated, order preserved)."""
    return list(get_workspace_materials().keys())


def page_grid_init(key_prefix=""):
    """Render grid inputs, return (x_min, x_max, y_min, y_max, nx, ny)."""
    p = f"{key_prefix}_" if key_prefix else ""
    st.caption("网格参数：")
    x_min = st.number_input("start x (µm)", value=-1.1, format="%.2f", key=f"{p}xmin")
    x_max = st.number_input("end   x (µm)", value=1.1, format="%.2f", key=f"{p}xmax")
    y_min = st.number_input("start y (µm)", value=-1.1, format="%.2f", key=f"{p}ymin")
    y_max = st.number_input("end   y (µm)", value=1.1, format="%.2f", key=f"{p}ymax")
    nx = st.number_input("nx", value=100, min_value=2, max_value=256, step=1, key=f"{p}nx")
    ny = st.number_input("ny", value=100, min_value=2, max_value=256, step=1, key=f"{p}ny")
    return x_min, x_max, y_min, y_max, nx, ny


def page_plane_wave_init(key_prefix="pw"):
    """Render plane wave params, return (wavelength, theta_deg, phi_deg)."""
    st.caption("光束参数：")
    wavelength = st.number_input("Wavelength (µm)", value=0.11, format="%.4f", key=f"{key_prefix}_wl")
    theta_deg = st.number_input("θ (deg)", min_value=-89.0, max_value=89.0, value=10.0, step=0.1, key=f"{key_prefix}_theta")
    phi_deg = st.number_input("φ (deg)", min_value=-180.0, max_value=180.0, value=30.0, step=0.1, key=f"{key_prefix}_phi")
    return wavelength, theta_deg, phi_deg


def page_z_ratio_wave_init(key_prefix="zr"):
    """Render z-ratio wave params, return (wavelength, z_ratio)."""
    st.caption("光束参数：")
    wavelength = st.number_input("Wavelength (µm)", value=1.1, format="%.4f", key=f"{key_prefix}_wl")
    z_ratio = st.number_input("z ratio (z = z ratio × wavelength)", value=0.25, format="%.2f", key=f"{key_prefix}_zr")
    return wavelength, z_ratio


def page_quadratic_wave_init(key_prefix="qw"):
    return page_z_ratio_wave_init(key_prefix)


def page_spherical_wave_init(key_prefix="sw"):
    return page_z_ratio_wave_init(key_prefix)


def page_flat_top_init(key_prefix="ft"):
    """Render flat-top params; return (mode, fraction, r, order, rx, ry, order_x, order_y)."""
    st.caption("光束参数：")
    mode = st.radio("Mode", ["Circular", "Rectangular"], key=f"{key_prefix}_mode", horizontal=True)
    fraction = st.number_input(
        "边缘幅度比", value=0.5, min_value=1e-6, max_value=1 - 1e-6, format="%.2f", key=f"{key_prefix}_frac"
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
    st.caption("光束参数：")
    wavelength = st.number_input("Wavelength (µm)", value=0.5, format="%.4f", key=f"{key_prefix}_wl")
    z_ratio = st.number_input("z ratio (z = z ratio × wavelength)", value=0.25, format="%.2f", key=f"{key_prefix}_zr")
    wx0 = st.number_input("wx0 (µm)", value=1.0, min_value=0.01, format="%.2f", key=f"{key_prefix}_wx0")
    wy0 = st.number_input("wy0 (µm)", value=1.0, min_value=0.01, format="%.2f", key=f"{key_prefix}_wy0")
    m = st.number_input("m", value=3, min_value=0, step=1, key=f"{key_prefix}_m")
    n = st.number_input("n", value=3, min_value=0, step=1, key=f"{key_prefix}_n")
    return wavelength, m, n, z_ratio*wavelength, wx0, wy0


def page_laguerre_gaussian_init(key_prefix="lg"):
    """Render Laguerre-Gaussian params, return (wavelength, p, l, z, w0)."""
    st.caption("光束参数：")
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


def get_z_ratio_wave_params_from_session(key_prefix):
    return (float(_p(key_prefix, "wl", 1.1)), float(_p(key_prefix, "zr", 0.25)))


def get_quadratic_wave_params_from_session(key_prefix):
    return get_z_ratio_wave_params_from_session(key_prefix)


def get_spherical_wave_params_from_session(key_prefix):
    return get_z_ratio_wave_params_from_session(key_prefix)


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


# 开发者可选：仅展示部分模板 id；None 表示展示 filmstack_templates.json 中全部条目。
FILMSTACK_TEMPLATE_FILTER: frozenset[str] | None = None


def get_filmstack_templates():
    from template_config import load_all_templates

    all_templates = load_all_templates()
    if FILMSTACK_TEMPLATE_FILTER is None:
        return all_templates
    allowed = FILMSTACK_TEMPLATE_FILTER
    return tuple(t for t in all_templates if t.preset.id in allowed)


def build_filmstack_preset_catalog():
    from template_config import build_preset_catalog, default_preset_id

    templates = get_filmstack_templates()
    if not templates:
        raise ValueError("FILMSTACK_TEMPLATE_FILTER excluded all templates")
    preset_id = default_preset_id()
    valid_ids = {t.preset.id for t in templates}
    if preset_id not in valid_ids:
        preset_id = templates[0].preset.id
    return build_preset_catalog(templates, default_preset_id=preset_id)


def get_filmstack_template_by_id():
    from template_config import template_by_id

    return template_by_id(get_filmstack_templates())


def get_default_material_path_keys() -> list[list[str]]:
    import simulation_database_parser as sdp
    from template_config import aggregate_material_path_keys

    base: list[list[str]] = [
        ["rii", "materials", "other", "mixed_gases", "air_Ciddor.yml"],
        *list(sdp.DEFAULT_RII_FILMSTACK_MATERIAL_PATHS.values()),
        ["rii", "materials", "specs", "schott", "optical", "N-BK7.yml"],
        ["rii", "materials", "main", "MgF2", "MgF2_Dodge-o.yml"],
        ["rii", "materials", "main", "TiO2", "TiO2_Jolivet-anatase.yml"],
        ["og", "materials", "oxides", "ITO", "ito.yml"],
        ["og", "materials", "small_molecules", "NPD.yml"],
        ["og", "materials", "small_molecules", "Alq3.yml"],
        ["og", "materials", "small_molecules", "TPBi.yml"],
        ["rii", "materials", "main", "LiF.yml"],
        ["og", "materials", "metal", "Al", "std.yml"],
    ]
    seen = {tuple(p) for p in base}
    for path in aggregate_material_path_keys(get_filmstack_templates()):
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            base.append(path)
    return base


def build_materials_db_from_path_keys(
    path_keys_list: Iterable[list[str]],
    *,
    sim_db: Any | None = None,
) -> Dict[str, Any]:
    """Load ``material_s`` objects for the given database path keys."""
    import simulation_database_parser as sdp
    from simulation_database.database_precompiling import (
        get_precompiled_leaf_object,
        load_or_build_database_index,
    )
    from simulation_database.database_ui import object_unique_name

    db = sim_db if sim_db is not None else sdp.get_simulation_database(init=True)
    load_or_build_database_index(db)
    out: Dict[str, Any] = {}
    for path_keys in path_keys_list:
        obj = get_precompiled_leaf_object(list(path_keys))
        out[object_unique_name(obj)] = obj
    return out


def build_default_materials_db(*, sim_db: Any | None = None) -> Dict[str, Any]:
    return build_materials_db_from_path_keys(get_default_material_path_keys(), sim_db=sim_db)


def get_required_default_material_names() -> frozenset[str]:
    from template_config import aggregate_required_material_names

    legacy = frozenset(
        {
            "air_Ciddor",
            "SiO2_Arosa",
            "Ta2O5_Cheikh-amorphous-3.28-8-450",
            "Si_Aspnes",
            "N-BK7",
            "MgF2_Dodge-o",
            "TiO2_Jolivet-anatase",
            "LiF",
            "ito",
            "NPD",
            "Alq3",
            "TPBi",
            "std",
        }
    )
    return legacy | aggregate_required_material_names(get_filmstack_templates())


def render_filmstack_host(*, render_page, PageContext) -> None:
    """Shared bootstrap for filmstack simulation / optimization host pages."""
    from simulation_database.workspace import ensure_sim_workspace_ui, get_workspace_materials
    import streamlit as st

    from toykits_config import resolve_filmstack_initial_defaults

    ensure_sim_workspace_ui()
    materials_db = get_workspace_materials()
    catalog = st.session_state.get("_filmstack_preset_catalog") or build_filmstack_preset_catalog()
    template_map = st.session_state.get("_filmstack_template_by_id") or get_filmstack_template_by_id()
    initial = st.session_state.get("_filmstack_initial_defaults") or resolve_filmstack_initial_defaults(
        catalog.valid_preset_ids,
        template_by_id=template_map,
    )
    render_page(
        context=PageContext(
            get_materials_db=get_workspace_materials,
            preset_catalog=catalog,
            template_by_id=template_map,
            recommended_wl_from=initial.wl_from_um,
            recommended_wl_to=initial.wl_to_um,
            initial_preset_id=initial.preset_id,
            initial_formula=initial.formula,
            tokens_path=HOST_DESIGN_TOKENS_PATH,
        ),
        materials_db=materials_db,
    )


def render_beam_compute_fragment(
    *,
    key_prefix: str,
    page_key: str,
    title_prefix: str,
    mat_filename: str,
    get_beam_params,
    compute_field,
) -> None:
    """Shared @st.fragment compute/result block for Gaussian optics beam pages."""

    @st.fragment
    def compute_and_result():
        x_min, x_max, y_min, y_max, nx, ny = get_grid_params_from_session(key_prefix)
        beam_params = get_beam_params(key_prefix)
        if st.button("▶️ 计算", width="stretch", key=f"{key_prefix}_btn"):
            try:
                start_xy = [x_min, y_min]
                end_xy = [x_max, y_max]
                shape_xy = [nx, ny]
                field, meta = compute_field(beam_params, start_xy, end_xy, shape_xy)
                from core import show_complex_plot

                fig = show_complex_plot(field, meta, title_prefix=title_prefix)
                st.session_state["beams_result_cache"][page_key] = {
                    "field": field,
                    "meta": meta,
                    "fig": fig,
                }
            except Exception as e:
                st.error(str(e))
        st.divider()
        st.subheader("结果")
        cache = st.session_state.get("beams_result_cache", {})
        if page_key in cache:
            entry = cache[page_key]
            pyplot_fixed_width(entry["fig"])
            import numpy as np

            f = entry["field"]
            st.caption(f"Shape {f.shape}, max |U| = {float(np.max(np.abs(f))):.4e}")
            mat_bytes = build_beam_result_mat(f, entry["meta"])
            save_result_mat_button(mat_bytes, mat_filename, f"{key_prefix}_save_mat")

    compute_and_result()
