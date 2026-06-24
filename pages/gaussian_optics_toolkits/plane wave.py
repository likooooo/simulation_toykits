import streamlit as st
from common import (
    ensure_beams_session_state,
    page_key_from_file,
    page_grid_init,
    page_plane_wave_init,
    get_plane_wave_params_from_session,
    render_beam_compute_fragment,
)
from core import compute_plane_wave

ensure_beams_session_state()
PAGE_KEY = page_key_from_file(__file__)
KEY_PREFIX = "pw"

st.set_page_config(page_title="Plane Wave", layout="wide")
st.header("Plane Wave")

st.subheader("Parameters")
col_grid, col_beam = st.columns(2)
with col_grid:
    page_grid_init(KEY_PREFIX)
with col_beam:
    page_plane_wave_init(KEY_PREFIX)

render_beam_compute_fragment(
    key_prefix=KEY_PREFIX,
    page_key=PAGE_KEY,
    title_prefix="Plane Wave",
    mat_filename="plane_wave_result.mat",
    get_beam_params=get_plane_wave_params_from_session,
    compute_field=lambda params, start, end, shape: compute_plane_wave(
        params[0], params[1], params[2], start, end, shape
    ),
)
