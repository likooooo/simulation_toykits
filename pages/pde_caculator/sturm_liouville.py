"""Sturm-Liouville PDE calculator page: axes config, MATLAB upload, formula, compute, download."""

import streamlit as st
from common import render_table_editor, pyplot_fixed_width, save_result_mat_button
from core import sturm_liouville

# 基础配置
BC_OPTIONS = ["periodic", "dirichlet", "neumann"]

def ensure_pde_sl_session_state():
    if "pde_sl_axes" not in st.session_state:
        st.session_state["pde_sl_axes"] = []
    if "pde_sl_mat_data" not in st.session_state:
        st.session_state["pde_sl_mat_data"] = None
    if "pde_sl_result" not in st.session_state:
        st.session_state["pde_sl_result"] = None
    if "pde_sl_pending_delete" not in st.session_state:
        st.session_state["pde_sl_pending_delete"] = None

def default_axis(dim_index: int):
    return {
        "n_points": 32,
        "length": 1.0,
        "bc_from": "dirichlet",
        "bc_to": "dirichlet",
        "p": 1.0,
        "q": 0.0,
    }

# --- 核心优化：回调函数处理联动，消除抖动 ---
def update_bc_callback(idx, side):
    """当边界条件改变时，立即同步数据，避免在渲染期间修改数据导致抖动"""
    axes_list = st.session_state["pde_sl_axes"]
    key_from = f"pde_sl_bcf_{idx}"
    key_to = f"pde_sl_bct_{idx}"
    
    val_from = st.session_state[key_from]
    val_to = st.session_state[key_to]

    if side == "from" and val_from == "periodic":
        st.session_state[key_to] = "periodic"
        axes_list[idx]["bc_from"] = "periodic"
        axes_list[idx]["bc_to"] = "periodic"
    elif side == "to" and val_to == "periodic":
        st.session_state[key_from] = "periodic"
        axes_list[idx]["bc_from"] = "periodic"
        axes_list[idx]["bc_to"] = "periodic"
    else:
        axes_list[idx]["bc_from"] = val_from
        axes_list[idx]["bc_to"] = val_to

def set_pending_delete(index: int):
    st.session_state["pde_sl_pending_delete"] = index
    st.rerun()

# --- 页面初始化 ---
st.set_page_config(page_title="Sturm-Liouville PDE", layout="wide")
ensure_pde_sl_session_state()

# 处理待删除任务
idx_del = st.session_state.get("pde_sl_pending_delete")
if idx_del is not None:
    axes_list = st.session_state["pde_sl_axes"]
    if 0 <= idx_del < len(axes_list):
        axes_list.pop(idx_del)
    st.session_state["pde_sl_pending_delete"] = None
    st.rerun()

st.header("Sturm-Liouville PDE 计算器")

axes: list = st.session_state["pde_sl_axes"]

# --- Upload MATLAB (v7) ---
st.subheader("上传 MATLAB 数据 (v7 格式)")
uploaded = st.file_uploader("选择 .mat 文件", type=["mat"], key="pde_sl_upload")
if uploaded is not None:
    try:
        data = sturm_liouville.load_mat_v7(uploaded.getvalue())
        st.session_state["pde_sl_mat_data"] = data
        ref_shape = data.get("u", data.get("f", data.get("analytical"))).shape
        ndim = len(ref_shape)
        
        # 自动匹配维度
        if len(st.session_state["pde_sl_axes"]) != ndim:
            st.session_state["pde_sl_axes"] = [
                { **default_axis(i), "n_points": int(ref_shape[i]) }
                for i in range(ndim)
            ]
            st.rerun()
        st.success(f"已加载数据，维度: {ref_shape}")
    except Exception as e:
        st.error(f"文件读取失败: {e}")
        st.session_state["pde_sl_mat_data"] = None

# --- Axes config（与 films / material 统一的表格式 UI）---
st.subheader("坐标轴配置")


def _pde_render_row(i, item, cols):
    with cols[0]:
        item["n_points"] = int(st.number_input("网格点数", min_value=2, value=item["n_points"], key=f"pde_sl_n_{i}", label_visibility="collapsed"))
    with cols[1]:
        item["length"] = float(st.number_input("域长", value=float(item["length"]), format="%.4f", key=f"pde_sl_L_{i}", label_visibility="collapsed"))
    with cols[2]:
        st.selectbox(
            "起点边界", BC_OPTIONS,
            index=BC_OPTIONS.index(item["bc_from"]),
            key=f"pde_sl_bcf_{i}",
            on_change=update_bc_callback,
            args=(i, "from"),
            label_visibility="collapsed",
        )
    with cols[3]:
        st.selectbox(
            "终点边界", BC_OPTIONS,
            index=BC_OPTIONS.index(item["bc_to"]),
            key=f"pde_sl_bct_{i}",
            on_change=update_bc_callback,
            args=(i, "to"),
            label_visibility="collapsed",
        )
    with cols[4]:
        item["p"] = float(st.number_input("p", value=float(item["p"]), format="%.4f", key=f"pde_sl_p_{i}", label_visibility="collapsed"))
    with cols[5]:
        item["q"] = float(st.number_input("q", value=float(item["q"]), format="%.4f", key=f"pde_sl_q_{i}", label_visibility="collapsed"))


def _pde_on_add():
    axes.append(default_axis(len(axes)))
    st.rerun()


def _pde_on_clear():
    st.session_state["pde_sl_axes"] = []
    st.rerun()


render_table_editor(
    key_prefix="pde_sl_axes",
    columns=[
        {"label": "网格点数", "width": 1},
        {"label": "域长", "width": 1},
        {"label": "起点边界", "width": 1.2},
        {"label": "终点边界", "width": 1.2},
        {"label": "p", "width": 0.8},
        {"label": "q", "width": 0.8},
    ],
    items=axes,
    render_row=_pde_render_row,
    on_add=_pde_on_add,
    on_clear=_pde_on_clear,
    on_delete=set_pending_delete,
    add_label="➕ 添加坐标轴",
    clear_label="🗑️ 清空",
    delete_label="删除",
)

# --- 公式、计算与结果（放入 fragment，点击计算时仅此块重跑，避免整页控件抖动）---
@st.fragment
def formula_compute_and_result():
    axes = st.session_state["pde_sl_axes"]
    mat_data = st.session_state.get("pde_sl_mat_data")
    st.subheader("公式预览")
    formula_md = sturm_liouville.sl_formula_markdown(axes, has_f=bool(mat_data and "f" in mat_data))
    st.markdown(formula_md)

    if st.button("▶️ 计算", width="stretch", key="pde_sl_compute"):
        if not axes:
            st.error("请至少添加一个坐标轴。")
        elif mat_data is None:
            st.error("请先上传数据。")
        else:
            try:
                result = sturm_liouville.run_sturm_liouville(axes, mat_data)
                st.session_state["pde_sl_result"] = result
                st.success("计算完成！")
            except Exception as e:
                st.error(f"计算出错: {e}")

    res = st.session_state.get("pde_sl_result")
    if res:
        st.subheader("结果可视化")
        fig = sturm_liouville.plot_result_and_error(res["result"], res.get("error"), input_field=res.get("input_field"))
        pyplot_fixed_width(fig)
        mat_bytes = sturm_liouville.build_result_mat(res["result"], res.get("error"))
        save_result_mat_button(mat_bytes, "sl_result.mat", "pde_sl_save_mat")
    else:
        st.caption("上传数据并配置坐标轴后，点击「▶️ 计算」得到结果。")

            

formula_compute_and_result()