import numpy as np
import streamlit as st
from common import ensure_beams_session_state, page_key_from_file, page_grid_init, page_flat_top_init
from core import compute_flat_top_rectangular, compute_flat_top_circular, show_complex_plot

ensure_beams_session_state()
PAGE_KEY = page_key_from_file(__file__)
KEY_PREFIX = "ft"

st.set_page_config(page_title="Flat-Top Beam", layout="wide")
st.header("Flat-Top Beam")

st.subheader("Parameters")
col_grid, col_beam = st.columns(2)
with col_grid:
    x_min, x_max, y_min, y_max, nx, ny = page_grid_init(KEY_PREFIX)
with col_beam:
    mode, fraction, r, order, rx, ry, order_x, order_y = page_flat_top_init(KEY_PREFIX)

def calculate():
    try:
        start_xy = [x_min, y_min]
        end_xy = [x_max, y_max]
        shape_xy = [nx, ny]
        if mode == "Circular":
            field, meta = compute_flat_top_circular(
                r, fraction, order, start_xy, end_xy, shape_xy
            )
        else:
            field, meta = compute_flat_top_rectangular(
                rx, ry, fraction, order_x, order_y, start_xy, end_xy, shape_xy
            )
        fig = show_complex_plot(field, meta, title_prefix="Flat-Top Beam")
        st.session_state["beams_result_cache"][PAGE_KEY] = {
            "field": field,
            "meta": meta,
            "fig": fig,
        }
        st.rerun()
    except Exception as e:
        st.error(str(e))

if st.button("▶️ 计算", width="stretch", key=f"{KEY_PREFIX}_btn"):
    calculate()

st.divider()
st.subheader("Result")
cache = st.session_state.get("beams_result_cache", {})
if PAGE_KEY in cache:
    entry = cache[PAGE_KEY]
    st.pyplot(entry["fig"])
    f = entry["field"]
    st.caption(f"Shape {f.shape}, max |U| = {float(np.max(np.abs(f))):.4e}")
else:
    calculate()
