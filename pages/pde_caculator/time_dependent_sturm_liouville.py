"""Time-dependent Sturm-Liouville 波动方程：基于 cache_pool 与 sl_web_factory。"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from common import save_result_mat_button, video_fixed_width
from core import sturm_liouville
from core.pde_caculator_core import (
    get_array_by_key,
    get_ref_shape_from_data,
    sl_web_factory,
)

_visualizer_dir = Path(__file__).resolve().parent.parent.parent / "assets" / "lib" / "core_plugins"
if _visualizer_dir.is_dir() and str(_visualizer_dir) not in sys.path:
    sys.path.insert(0, str(_visualizer_dir))
import visualizer


def _frames_to_video_bytes_via_visualizer(frames, fps):
    if not frames:
        raise ValueError("frames is empty")
    fd, path = tempfile.mkstemp(suffix=".mp4")
    try:
        os.close(fd)
        visualizer.SAVE_TO_FILE = True
        visualizer.VIDEO_OUTPUT_PATH = path
        visualizer.VIDEO_FPS = fps
        visualizer.init_realtime_multi(1, 1)
        for fr in frames:
            plot_arrays = [
                sturm_liouville._to_real_array(fr, "real"),
                # sturm_liouville._to_real_array(fr, "imag"),
                # sturm_liouville._to_real_array(fr, "amplitude"),
                # sturm_liouville._to_real_array(fr, "phase"),
            ]
            visualizer.update_multi(plot_arrays)
        visualizer.reset_realtime_multi()
        with open(path, "rb") as f:
            return f.read()
    finally:
        visualizer.SAVE_TO_FILE = False
        visualizer.VIDEO_OUTPUT_PATH = None
        try:
            os.unlink(path)
        except OSError:
            pass


def _render_cache_section_tdsl(key_prefix: str, options: list, keys: list, labels: list, choices: list) -> None:
    """TDSL 选择输入：u(0) + u'(0)(可选)。"""
    col_u0, col_ut0 = st.columns(2)
    with col_u0:
        idx_u0 = 0
        if options:
            u0_current = st.session_state.get(f"{key_prefix}_u0_key", "")
            if u0_current in keys:
                idx_u0 = keys.index(u0_current)
            st.selectbox(
                "u(0)",
                range(len(choices)),
                format_func=lambda i: choices[i],
                index=idx_u0,
                key=f"{key_prefix}_u0_select",
            )
    with col_ut0:
        ut0_choices = ["（无）"] + labels
        ut0_idx = 0
        ut0_current = st.session_state.get(f"{key_prefix}_ut0_key", "")
        if ut0_current in keys:
            ut0_idx = keys.index(ut0_current) + 1
        st.selectbox(
            "u'(0)（可选项, 默认为0）",
            range(len(ut0_choices)),
            format_func=lambda i: ut0_choices[i],
            index=ut0_idx,
            key=f"{key_prefix}_ut0_select",
        )
    if options and 0 <= st.session_state.get(f"{key_prefix}_u0_select", 0) < len(keys):
        st.session_state[f"{key_prefix}_u0_key"] = keys[st.session_state[f"{key_prefix}_u0_select"]]
    ut0_sel = st.session_state.get(f"{key_prefix}_ut0_select", 0)
    if 1 <= ut0_sel <= len(keys):
        st.session_state[f"{key_prefix}_ut0_key"] = keys[ut0_sel - 1]
    else:
        st.session_state[f"{key_prefix}_ut0_key"] = ""


def _get_effective_data_tdsl(key_prefix: str):
    """从 session state 按 TDSL 选择拼出 load_mat_v7 同结构 data。"""
    u0_key = st.session_state.get(f"{key_prefix}_u0_key", "")
    ut0_key = st.session_state.get(f"{key_prefix}_ut0_key", "")
    if not u0_key:
        return None
    u0_arr = get_array_by_key(st.session_state, u0_key)
    if u0_arr is None:
        return None
    data = {"u": u0_arr}
    if ut0_key:
        ut0_arr = get_array_by_key(st.session_state, ut0_key)
        if ut0_arr is not None:
            data["ut"] = ut0_arr
    return data


st.set_page_config(page_title="Time-Dependent Sturm-Liouville", layout="wide")
f = sl_web_factory("pde_tdsl", _get_effective_data_tdsl)
f.run_start()

st.header("Time-Dependent Sturm-Liouville 波动方程")

f.init_matlab_upload(sturm_liouville.load_mat_v7)
f.init_cache_section(_render_cache_section_tdsl)
f.init_axes_config()

st.subheader("时间范围与波速")
col_t1, col_t2, col_dt, col_c = st.columns(4)
with col_t1:
    st.number_input("t 起始", value=0.0, format="%.4f", key="pde_tdsl_t_start")
with col_t2:
    st.number_input("t 结束", value=1.0, format="%.4f", key="pde_tdsl_t_end")
with col_dt:
    st.number_input("时间步长 dt", value=0.05, format="%.4f", min_value=1e-6, key="pde_tdsl_dt")
with col_c:
    st.number_input("波速 c", value=1.0, format="%.4f", min_value=1e-6, key="pde_tdsl_c")


def _formula_md(axes: list, data: dict | None) -> str:
    part1 = "**波动方程** $u_{tt} = c^2 \\mathcal{L} u$，其中 $\\mathcal{L}$ 为多轴 Sturm-Liouville 算子（由坐标轴配置确定）。"
    part2 = sturm_liouville.sl_formula_markdown(axes, has_f=False)
    return part1 + "\n\n" + part2


def _on_compute() -> None:
    import numpy as np
    axes = st.session_state.get(f._axes_key) or []
    data = f.get_effective_data()
    u0 = data.get("u") if data else None
    ut0 = data.get("ut") if data else None
    if not axes:
        st.error("请至少添加一个坐标轴。")
        return
    if u0 is None:
        st.error("请先上传 .mat 或从缓存选择 u(0)。")
        return
    t_start = float(st.session_state.get("pde_tdsl_t_start", 0.0))
    t_end = float(st.session_state.get("pde_tdsl_t_end", 1.0))
    dt = float(st.session_state.get("pde_tdsl_dt", 0.05))
    wave_speed = float(st.session_state.get("pde_tdsl_c", 1.0))
    u0_arr = np.asarray(u0, dtype=np.complex128)
    ut0_arr = np.asarray(ut0, dtype=np.complex128) if ut0 is not None else None
    ref_shape = u0_arr.shape
    if len(axes) != len(ref_shape):
        if f.refresh_axes_if_needed():
            return
    try:
        result = sturm_liouville.run_time_dependent_sturm_liouville(
            axes, u0_arr, ut0_arr, wave_speed, t_start, t_end, dt
        )
        st.session_state[f._result_key] = result
        st.success("计算完成！")
        st.rerun()
    except Exception as e:
        st.error(f"计算出错: {e}")


def _render_result(res: dict) -> None:
    st.subheader("结果视频")
    n_frames = len(res["frames"])
    fps = max(1, round(n_frames / 5.0))
    try:
        video_bytes = _frames_to_video_bytes_via_visualizer(res["frames"], fps=fps)
        video_fixed_width(video_bytes)
        save_result_mat_button(video_bytes, "tdsl_wave.mp4", "pde_tdsl_dl_video")
    except Exception as e:
        st.error(f"生成视频失败: {e}")
    st.caption(
        f"共 {n_frames} 帧，时间 t ∈ [{res['t_vals'].min():.4f}, {res['t_vals'].max():.4f}]，"
        f"fps={fps}（约 {n_frames/max(1,fps):.1f}s）"
    )


@st.fragment
def _formula_block():
    f.init_formula_preview(_formula_md)
    f.init_compute_and_result(
        compute_button_label="▶️ 计算并生成视频",
        compute_key="pde_tdsl_compute",
        on_compute=_on_compute,
        render_result_section=_render_result,
        empty_caption="配置 u(0)、坐标轴与时间范围后，点击「▶️ 计算并生成视频」。",
    )


_formula_block()
