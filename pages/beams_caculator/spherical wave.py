import numpy as np
import streamlit as st
from common import (
    ensure_beams_session_state,
    page_key_from_file,
    page_grid_init,
    page_spherical_wave_init,
    get_grid_params_from_session,
    get_spherical_wave_params_from_session,
    pyplot_fixed_width,
    save_result_mat_button,
    build_beam_result_mat,
)
from core import compute_spherical_wave, show_complex_plot

ensure_beams_session_state()
PAGE_KEY = page_key_from_file(__file__)
KEY_PREFIX = "sw"

st.set_page_config(page_title="Spherical Wave", layout="wide")
st.header("Spherical Wave")

st.subheader("Parameters")
col_grid, col_beam = st.columns(2)
with col_grid:
    x_min, x_max, y_min, y_max, nx, ny = page_grid_init(KEY_PREFIX)
with col_beam:
    wavelength, z_ratio = page_spherical_wave_init(KEY_PREFIX)

@st.fragment
def compute_and_result():
    x_min, x_max, y_min, y_max, nx, ny = get_grid_params_from_session(KEY_PREFIX)
    wavelength, z_ratio = get_spherical_wave_params_from_session(KEY_PREFIX)
    if st.button("▶️ 计算", width="stretch", key=f"{KEY_PREFIX}_btn"):
        try:
            start_xy = [x_min, y_min]
            end_xy = [x_max, y_max]
            shape_xy = [nx, ny]
            field, meta = compute_spherical_wave(
                wavelength, z_ratio, start_xy, end_xy, shape_xy
            )
            fig = show_complex_plot(field, meta, title_prefix="Spherical Wave")
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
        save_result_mat_button(mat_bytes, "spherical_wave_result.mat", f"{KEY_PREFIX}_save_mat")

compute_and_result()
