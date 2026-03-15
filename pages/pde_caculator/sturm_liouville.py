"""Sturm-Liouville PDE 计算器：基于 cache_pool 与 sl_web_factory。"""

import streamlit as st
from common import pyplot_fixed_width, save_result_mat_button
from core import sturm_liouville, show_complex_plot
from core.beams_plot import show_complex_plot_amp_phase
from core.pde_caculator_core import (
    get_array_by_key,
    get_ref_shape_from_data,
    sl_web_factory,
    put_result_into_cache_pool,
)


def _pde_display_2d_and_meta(arr, axes: list) -> tuple:
    import numpy as np
    shape = arr.shape
    ndim = len(shape)

    def _dx(i):
        if i < len(axes) and axes[i]:
            return float(axes[i]["length"]) / max(1, int(axes[i]["n_points"]))
        return 1.0

    if ndim == 1:
        arr_2d = np.asarray(arr).reshape(1, -1)
        meta = {"nx": shape[0], "ny": 1, "dx": _dx(0), "dy": 1.0}
    elif ndim == 2:
        arr_2d = np.asarray(arr)
        meta = {"nx": shape[0], "ny": shape[1], "dx": _dx(0), "dy": _dx(1)}
    else:
        mid = shape[-1] // 2
        arr_2d = np.asarray(arr)[..., mid]
        meta = {"nx": shape[0], "ny": shape[1], "dx": _dx(0), "dy": _dx(1)}
    return arr_2d, meta


def _render_cache_section_sl(key_prefix: str, options: list, keys: list, labels: list, choices: list) -> None:
    """SL 选择输入：input + 含义(f/u) + analytical(可选)。"""
    col_in, col_role = st.columns([3, 1])
    with col_in:
        idx = 0
        if options:
            current = st.session_state.get(f"{key_prefix}_input_key", "")
            if current in keys:
                idx = keys.index(current)
            st.selectbox(
                "input",
                range(len(choices)),
                format_func=lambda i: choices[i],
                index=idx,
                key=f"{key_prefix}_input_select",
            )
    with col_role:
        # index 0 -> "f", 1 -> "u"，与显示文案解耦
        _ROLE_OPTIONS = ["f (求解方程 Lu=f)", "u (求解算子 L 作用于 u)"]
        st.radio(
            "input 类型",
            range(len(_ROLE_OPTIONS)),
            format_func=lambda i: _ROLE_OPTIONS[i],
            index=st.session_state.get(f"{key_prefix}_cache_role", 0),
            horizontal=True,
            key=f"{key_prefix}_cache_role",
        )
    if options and 0 <= st.session_state.get(f"{key_prefix}_input_select", 0) < len(keys):
        st.session_state[f"{key_prefix}_input_key"] = keys[st.session_state[f"{key_prefix}_input_select"]]

    analytical_choices = ["（无）"] + labels
    a_idx = 0
    a_current = st.session_state.get(f"{key_prefix}_analytical_key", "")
    if a_current in keys:
        a_idx = keys.index(a_current) + 1
    st.selectbox(
        "analytical（可选项, 用于比对结果）",
        range(len(analytical_choices)),
        format_func=lambda i: analytical_choices[i],
        index=a_idx,
        key=f"{key_prefix}_analytical_select",
    )
    a_sel = st.session_state.get(f"{key_prefix}_analytical_select", 0)
    if 1 <= a_sel < len(analytical_choices):
        st.session_state[f"{key_prefix}_analytical_key"] = keys[a_sel - 1]
    else:
        st.session_state[f"{key_prefix}_analytical_key"] = ""


# SL input 类型：radio index 0 -> "f", 1 -> "u"
_ROLE_INDEX_TO_KEY = ("f", "u")


def _get_effective_data_sl(key_prefix: str):
    """从 session state 按 SL 选择拼出 load_mat_v7 同结构 data。"""
    inp_key = st.session_state.get(f"{key_prefix}_input_key", "")
    role_index = st.session_state.get(f"{key_prefix}_cache_role", 0)
    role = _ROLE_INDEX_TO_KEY[role_index] if 0 <= role_index < len(_ROLE_INDEX_TO_KEY) else "f"
    ana_key = st.session_state.get(f"{key_prefix}_analytical_key", "")
    if not inp_key:
        return None
    arr = get_array_by_key(st.session_state, inp_key)
    if arr is None:
        return None
    data = {role: arr}
    if ana_key:
        ana_arr = get_array_by_key(st.session_state, ana_key)
        if ana_arr is not None:
            data["analytical"] = ana_arr
    return data


st.set_page_config(page_title="Sturm-Liouville PDE", layout="wide")
f = sl_web_factory("pde_sl", _get_effective_data_sl)
f.run_start()

st.header("Sturm-Liouville PDE 计算器")

f.init_matlab_upload(sturm_liouville.load_mat_v7)
f.init_cache_section(_render_cache_section_sl)
f.init_axes_config()


def _formula_md(axes: list, data: dict | None) -> str:
    return sturm_liouville.sl_formula_markdown(axes, has_f=bool(data and "f" in data))


def _on_compute() -> None:
    axes = st.session_state.get(f._axes_key) or []
    data = f.get_effective_data()
    if not axes:
        st.error("请至少添加一个坐标轴。")
        return
    if data is None:
        st.error("请先上传数据或从缓存选择一项作为 f 或 u。")
        return
    ref_shape = get_ref_shape_from_data(data)
    if ref_shape is not None and len(axes) != len(ref_shape):
        if f.refresh_axes_if_needed():
            return
    try:
        result = sturm_liouville.run_sturm_liouville(axes, data)
        st.session_state[f._result_key] = result
        put_result_into_cache_pool(st.session_state, result["result"], "PDE/Sturm Liouville output")
        st.success("计算完成！")
        st.rerun()
    except Exception as e:
        msg = str(e).strip() or type(e).__name__
        st.error(f"计算出错: {msg}")


def _render_result(res: dict) -> None:
    axes = st.session_state.get(f._axes_key) or []
    result_arr = res["result"]
    input_arr = res.get("input_field")
    analytical_arr = res.get("analytical")
    slice_kw = {"index": result_arr.shape[-1] // 2} if result_arr.ndim >= 3 else {}
    st.subheader("结果可视化")
    if input_arr is not None:
        in_2d, meta = _pde_display_2d_and_meta(input_arr, axes)
        fig_in = show_complex_plot_amp_phase(in_2d, meta, title_prefix="Input")
        pyplot_fixed_width(fig_in)
    out_2d, meta = _pde_display_2d_and_meta(result_arr, axes)
    fig_out = show_complex_plot(out_2d, meta, title_prefix="Sturm-Liouville Output")
    pyplot_fixed_width(fig_out)
    if analytical_arr is not None:
        fig_err = sturm_liouville.plot_error_only(result_arr, analytical_arr, slice_kw)
        pyplot_fixed_width(fig_err)
    mat_bytes = sturm_liouville.build_result_mat(res["result"], res.get("error"))
    save_result_mat_button(mat_bytes, "sl_result.mat", "pde_sl_save_mat")


@st.fragment
def _formula_block():
    f.init_formula_preview(_formula_md)
    f.init_compute_and_result(
        compute_button_label="▶️ 计算",
        compute_key="pde_sl_compute",
        on_compute=_on_compute,
        render_result_section=_render_result,
        empty_caption="上传数据并配置坐标轴后，点击「▶️ 计算」得到结果。",
    )


_formula_block()
