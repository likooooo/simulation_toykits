from common import ensure_fresnel_session_state, init_materials_db
from core import refractiveindex as ri
import streamlit as st
import pandas as pd
import os

from core.material_index import load_material_index_cached
from core.plots import build_nk_curve_figure

ensure_fresnel_session_state()

csv_path = os.path.join(ri.meterial_db_path, "materials_index.csv")

st.set_page_config(page_title="Fresnel caculator (select material)", layout="wide")
st.header("材料数据库")

if "selected_shelf" not in st.session_state:
    st.session_state.selected_shelf = "MAIN - simple inorganic materials"
if "selected_book" not in st.session_state:
    st.session_state.selected_book = "SiO2 (Silicon dioxide, Silica, Quartz)"
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Malitson 1965: n 0.21–6.7 µm"

df = load_material_index_cached(csv_path)
if df.empty:
    st.stop()

shelf_map = df[["shelf_name", "shelf_id"]].drop_duplicates().sort_values("shelf_name")
shelf_list = shelf_map["shelf_name"].tolist()
shelf_index = (
    shelf_list.index(st.session_state.selected_shelf)
    if st.session_state.selected_shelf in shelf_list
    else 0
)

selected_shelf_name = st.selectbox(
    "Shelf", options=shelf_list, index=shelf_index, key="shelf_box"
)
st.session_state.selected_shelf = selected_shelf_name
selected_shelf_id = shelf_map[shelf_map["shelf_name"] == selected_shelf_name]["shelf_id"].iloc[0]

book_df = df[df["shelf_id"] == selected_shelf_id]
book_map = book_df[["book_name", "book_id"]].drop_duplicates().sort_values("book_name")
book_list = book_map["book_name"].tolist()
book_index = (
    book_list.index(st.session_state.selected_book)
    if st.session_state.selected_book in book_list
    else 0
)

selected_book_name = st.selectbox(
    "Book", options=book_list, index=book_index, key="book_box"
)
st.session_state.selected_book = selected_book_name
selected_book_id = book_map[book_map["book_name"] == selected_book_name]["book_id"].iloc[0]

page_df = df[(df["shelf_id"] == selected_shelf_id) & (df["book_id"] == selected_book_id)]
page_map = page_df[["page_name", "page_id"]].sort_values("page_name")
page_list = page_map["page_name"].tolist()
page_index = (
    page_list.index(st.session_state.selected_page)
    if st.session_state.selected_page in page_list
    else 0
)

selected_page_name = st.selectbox(
    "Page", options=page_list, index=page_index, key="page_box"
)
st.session_state.selected_page = selected_page_name
selected_page_id = page_map[page_map["page_name"] == selected_page_name]["page_id"].iloc[0]


def add_material():
    mat_info = {
        "Shelf ID": selected_shelf_id,
        "Book ID": selected_book_id,
        "Page ID": selected_page_id,
        "Material Name": selected_book_name,
        "Data Source": selected_page_name,
    }
    st.session_state["materials_db"][selected_book_id] = mat_info
    st.toast(f"已添加: {selected_book_name}")


col_cfg1, col_cfg2 = st.columns(2)
with col_cfg1:
    st.button("➕ 添加", on_click=add_material, width="stretch")
with col_cfg2:
    if st.button("🗑️ 清空", width="stretch"):
        init_materials_db()
        st.rerun()

plot_shelf, plot_book, plot_page, plot_name = (
    selected_shelf_id,
    selected_book_id,
    selected_page_id,
    selected_book_name,
)

if st.session_state["materials_db"]:
    with st.expander("📊 材料列表", expanded=True):
        summary_df = pd.DataFrame.from_dict(
            st.session_state["materials_db"], orient="index"
        )
        selection_event = st.dataframe(
            summary_df[["Book ID", "Data Source", "Shelf ID"]],
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
        )
        if selection_event and selection_event.selection.rows:
            selected_idx = selection_event.selection.rows[0]
            row = summary_df.iloc[selected_idx]
            plot_shelf = row["Shelf ID"]
            plot_book = row["Book ID"]
            plot_page = row["Page ID"]
            plot_name = row["Material Name"]

if plot_book:
    try:
        wavelengths, n_vals, k_vals = ri.load_nk(plot_shelf, plot_book, plot_page)
        fig = build_nk_curve_figure(
            wavelengths, n_vals, k_vals, title=plot_name
        )
        st.plotly_chart(
            fig,
            config={"displayModeBar": True, "displaylogo": False},
        )

        export_df = pd.DataFrame({
            "Wavelength(um)": wavelengths,
            "n": n_vals,
            "k": k_vals,
        })
        csv_data = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 下载 CSV",
            data=csv_data,
            file_name=f"{plot_book}_{plot_page}.csv",
            mime="text/csv",
            width="stretch",
        )
    except Exception as e:
        st.error(f"数据获取失败: {e}")
