import streamlit as st
from common import (
    ensure_beams_session_state,
    page_key_from_file,
    page_grid_init,
    page_spherical_wave_init,
    get_spherical_wave_params_from_session,
    render_beam_compute_fragment,
)
from core import compute_spherical_wave

ensure_beams_session_state()
PAGE_KEY = page_key_from_file(__file__)
KEY_PREFIX = "sw"

st.set_page_config(page_title="Spherical Wave", layout="wide")
st.header("Spherical Wave")

st.subheader("Parameters")
col_grid, col_beam = st.columns(2)
with col_grid:
    page_grid_init(KEY_PREFIX)
with col_beam:
    page_spherical_wave_init(KEY_PREFIX)

render_beam_compute_fragment(
    key_prefix=KEY_PREFIX,
    page_key=PAGE_KEY,
    title_prefix="Spherical Wave",
    mat_filename="spherical_wave_result.mat",
    get_beam_params=get_spherical_wave_params_from_session,
    compute_field=lambda params, start, end, shape: compute_spherical_wave(
        params[0], params[1], start, end, shape
    ),
)
