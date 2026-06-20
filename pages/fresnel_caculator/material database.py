from common import ensure_fresnel_session_state, init_materials_db
import streamlit as st
import pandas as pd

from core import simulation_loader
from core.plots import build_nk_curve_figure
from core.simulation_database_ui import (
    current_leaf_query,
    descend_frame,
    ensure_simulation_database_initialized,
    material_nk_arrays,
    material_to_csv_bytes,
    new_nav_stack,
    read_leaf_material,
)

ensure_fresnel_session_state()
simulation_loader.ensure_artifacts_on_path()

st.set_page_config(page_title="Fresnel caculator (select material)", layout="wide")
st.header("材料数据库")

if "db_nav_stack" not in st.session_state:
    st.session_state["db_nav_stack"] = new_nav_stack()

sim_db = ensure_simulation_database_initialized(st.session_state["db_nav_stack"][0].sim_db)

prep_log = st.session_state.get("_sim_db_prepare_log")
if prep_log:
    with st.expander("材料库状态", expanded=False):
        st.code("\n".join(prep_log))

st.divider()

# --- 浏览导航 ---
stack = st.session_state["db_nav_stack"]
frame = stack[-1]
keys = frame.keys

st.caption(f"**{frame.breadcrumb()}**")
nav_col1, nav_col2, nav_col3 = st.columns([3, 1, 1])
with nav_col1:
    if keys:
        selected_key = st.selectbox(
            "当前层级",
            options=keys,
            index=min(frame.selected, len(keys) - 1),
            key="db_key_select",
        )
        frame.selected = keys.index(selected_key)
    else:
        st.info("当前层级为空，请稍候（材料库正在后台准备）或返回上级。")
        selected_key = None
with nav_col2:
    enter = st.button("进入 / 选中", width="stretch", disabled=not keys)
with nav_col3:
    if st.button("返回上级", width="stretch", disabled=len(stack) <= 1):
        stack.pop()
        st.rerun()

if enter and selected_key:
    is_leaf, msg = descend_frame(stack, selected_key)
    if msg:
        st.toast(msg)
    st.rerun()

leaf = current_leaf_query(stack)
preview_mat = None
if leaf is not None and frame.oghma is not None:
    try:
        preview_mat = read_leaf_material(frame.oghma, leaf)
    except Exception as e:
        st.error(f"读取材料失败: {e}")

act_col1, act_col2 = st.columns(2)
with act_col1:
    if st.button("➕ 添加当前材料", width="stretch", disabled=preview_mat is None):
        name = getattr(preview_mat, "name", None) or selected_key or "material"
        st.session_state["materials_db"][name] = preview_mat
        st.toast(f"已添加: {name}")
with act_col2:
    if st.button("🗑️ 清空材料列表", width="stretch"):
        init_materials_db()
        st.rerun()

# --- 已添加材料列表 ---
plot_mat = preview_mat
if st.session_state["materials_db"]:
    st.subheader("📊 材料列表")
    rows = []
    for name, mat in st.session_state["materials_db"].items():
        src = ""
        if hasattr(mat, "source_path"):
            try:
                src = str(mat.source_path())
            except Exception:
                src = str(getattr(mat, "source_path", ""))
        rows.append({"Material": name, "Source": src})
    summary_df = pd.DataFrame(rows)
    selection_event = st.dataframe(
        summary_df,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
    )
    if selection_event and selection_event.selection.rows:
        selected_idx = selection_event.selection.rows[0]
        name = summary_df.iloc[selected_idx]["Material"]
        plot_mat = st.session_state["materials_db"].get(name)

if plot_mat is not None:
    try:
        wavelengths, n_vals, k_vals = material_nk_arrays(plot_mat)
        title = getattr(plot_mat, "name", "material")
        fig = build_nk_curve_figure(wavelengths, n_vals, k_vals, title=title)
        st.plotly_chart(fig, config={"displayModeBar": True, "displaylogo": False})
        csv_data = material_to_csv_bytes(plot_mat)
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in title)
        st.download_button(
            label="📥 下载材料 nk CSV",
            data=csv_data,
            file_name=f"{safe_name}.csv",
            mime="text/csv",
            width="stretch",
        )
    except Exception as e:
        st.error(f"数据获取失败: {e}")
