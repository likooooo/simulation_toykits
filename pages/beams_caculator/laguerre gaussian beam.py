import numpy as np
import streamlit as st
from common import (
    ensure_beams_session_state,
    page_key_from_file,
    page_grid_init,
    page_laguerre_gaussian_init,
    get_grid_params_from_session,
    get_laguerre_gaussian_params_from_session,
    pyplot_fixed_width,
    save_result_mat_button,
    build_beam_result_mat,
)
from core import compute_laguerre_gaussian, show_complex_plot

ensure_beams_session_state()
PAGE_KEY = page_key_from_file(__file__)
KEY_PREFIX = "lg"

st.set_page_config(page_title="Laguerre-Gaussian Beam", layout="wide")
st.header("Laguerre-Gaussian Beam")

st.subheader("Parameters")
col_grid, col_beam = st.columns(2)
with col_grid:
    x_min, x_max, y_min, y_max, nx, ny = page_grid_init(KEY_PREFIX)
with col_beam:
    wavelength, p, l, z, w0 = page_laguerre_gaussian_init(KEY_PREFIX)

@st.fragment
def compute_and_result():
    x_min, x_max, y_min, y_max, nx, ny = get_grid_params_from_session(KEY_PREFIX)
    wavelength, p, l, z, w0 = get_laguerre_gaussian_params_from_session(KEY_PREFIX)
    if st.button("▶️ 计算", width="stretch", key=f"{KEY_PREFIX}_btn"):
        try:
            start_xy = [x_min, y_min]
            end_xy = [x_max, y_max]
            shape_xy = [nx, ny]
            field, meta = compute_laguerre_gaussian(
                p, l, wavelength, z, w0, start_xy, end_xy, shape_xy
            )
            fig = show_complex_plot(field, meta, title_prefix="Laguerre-Gaussian")
            st.session_state["beams_result_cache"][PAGE_KEY] = {
                "field": field,
                "meta": meta,
                "fig": fig,
            }
        except Exception as e:
            st.error(str(e))
    st.divider()
    st.subheader("Result")
    cache = st.session_state.get("beams_result_cache", {})
    if PAGE_KEY in cache:
        entry = cache[PAGE_KEY]
        pyplot_fixed_width(entry["fig"])
        f = entry["field"]
        st.caption(f"Shape {f.shape}, max |U| = {float(np.max(np.abs(f))):.4e}")
        mat_bytes = build_beam_result_mat(f, entry["meta"])
        save_result_mat_button(mat_bytes, "laguerre_gaussian_result.mat", f"{KEY_PREFIX}_save_mat")

compute_and_result()
