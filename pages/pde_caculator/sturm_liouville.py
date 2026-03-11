"""Sturm-Liouville PDE calculator page: axes config, MATLAB upload, formula, compute, download."""

import streamlit as st
from core import sturm_liouville

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


def sync_periodic(axes: list):
    """If one end is periodic, set the other to periodic."""
    for ax in axes:
        if ax.get("bc_from") == "periodic":
            ax["bc_to"] = "periodic"
        if ax.get("bc_to") == "periodic":
            ax["bc_from"] = "periodic"


def set_pending_delete(index: int):
    st.session_state["pde_sl_pending_delete"] = index


st.set_page_config(page_title="Sturm-Liouville PDE", layout="wide")
ensure_pde_sl_session_state()

# Process pending delete at start of run so it works without navigating away
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
        # Auto-refresh axes to match dimension
        current = st.session_state["pde_sl_axes"]
        if len(current) != ndim:
            st.session_state["pde_sl_axes"] = [
                {
                    **default_axis(i),
                    "n_points": int(ref_shape[i]),
                    "length": 1.0,
                }
                for i in range(ndim)
            ]
            st.rerun()
        st.success(f"已加载: u={'u' in data}, f={'f' in data}, analytical={'analytical' in data}, shape={ref_shape}")
    except Exception as e:
        st.error(str(e))
        st.session_state["pde_sl_mat_data"] = None

# --- Axes config ---
st.subheader("坐标轴配置")
col_add, _ = st.columns([1, 3])
with col_add:
    if st.button("➕ 添加坐标轴", key="pde_sl_add_axis"):
        axes.append(default_axis(len(axes)))
        st.rerun()

for i in range(len(axes)):
    sync_periodic(axes)  # So periodic on one end auto-switches the other before we render
    with st.expander(f"坐标轴 {i+1}", expanded=True):
        cols = st.columns(6)
        with cols[0]:
            axes[i]["n_points"] = int(st.number_input("网格点数", min_value=2, value=axes[i].get("n_points", 32), key=f"pde_sl_n_{i}"))
        with cols[1]:
            axes[i]["length"] = float(st.number_input("域长", value=float(axes[i].get("length", 1.0)), format="%.4f", key=f"pde_sl_L_{i}"))
        with cols[2]:
            bc_from = st.selectbox("起点边界", BC_OPTIONS, index=BC_OPTIONS.index(axes[i].get("bc_from", "dirichlet")), key=f"pde_sl_bcf_{i}")
            axes[i]["bc_from"] = bc_from
            if bc_from == "periodic":
                axes[i]["bc_to"] = "periodic"
        with cols[3]:
            bc_to = st.selectbox("终点边界", BC_OPTIONS, index=BC_OPTIONS.index(axes[i].get("bc_to", "dirichlet")), key=f"pde_sl_bct_{i}")
            axes[i]["bc_to"] = bc_to
            if bc_to == "periodic":
                axes[i]["bc_from"] = "periodic"
        with cols[4]:
            axes[i]["p"] = float(st.number_input("p", value=float(axes[i].get("p", 1.0)), format="%.4f", key=f"pde_sl_p_{i}"))
        with cols[5]:
            axes[i]["q"] = float(st.number_input("q", value=float(axes[i].get("q", 0.0)), format="%.4f", key=f"pde_sl_q_{i}"))
        sync_periodic(axes)
        st.button(
            "删除",
            key=f"pde_sl_del_{i}",
            on_click=set_pending_delete,
            args=(i,),
        )

sync_periodic(axes)

# --- Refresh formula ---
st.subheader("公式")
if st.button("刷新公式", key="pde_sl_refresh_formula"):
    st.rerun()
mat_data = st.session_state.get("pde_sl_mat_data")
has_f = bool(mat_data and "f" in mat_data)
formula_md = sturm_liouville.sl_formula_markdown(axes, has_f=has_f)
st.markdown(formula_md)

# --- Compute ---
st.subheader("计算")
if st.button("计算", key="pde_sl_compute"):
    if not axes:
        st.error("请至少添加一个坐标轴。")
    elif mat_data is None:
        st.error("请先上传包含 u 或 f 的 MATLAB 文件。")
    else:
        try:
            result = sturm_liouville.run_sturm_liouville(axes, mat_data)
            st.session_state["pde_sl_result"] = result
            st.success("计算完成。")
            st.rerun()
        except Exception as e:
            st.error(str(e))

# --- Result & plot ---
st.subheader("结果")
res = st.session_state.get("pde_sl_result")
if res is not None:
    fig = sturm_liouville.plot_result_and_error(
        res["result"],
        res.get("error"),
        input_field=res.get("input_field"),
    )
    st.pyplot(fig)
    # Download
    mat_bytes = sturm_liouville.build_result_mat(res["result"], res.get("error"))
    st.download_button(
        "下载结果为 MATLAB 文件",
        data=mat_bytes,
        file_name="sturm_liouville_result.mat",
        mime="application/octet-stream",
        key="pde_sl_download",
    )
else:
    st.info("上传数据并点击「计算」后显示结果。")
