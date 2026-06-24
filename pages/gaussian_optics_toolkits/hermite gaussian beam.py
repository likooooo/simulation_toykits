import streamlit as st
from common import (
    ensure_beams_session_state,
    page_key_from_file,
    page_grid_init,
    page_hermite_gaussian_init,
    get_hermite_gaussian_params_from_session,
    render_beam_compute_fragment,
)
from core import compute_hermite_gaussian

ensure_beams_session_state()
PAGE_KEY = page_key_from_file(__file__)
KEY_PREFIX = "hg"

st.set_page_config(page_title="Hermite-Gaussian Beam", layout="wide")
st.header("Hermite-Gaussian Beam")

st.subheader("Parameters")
col_grid, col_beam = st.columns(2)
with col_grid:
    page_grid_init(KEY_PREFIX)
with col_beam:
    page_hermite_gaussian_init(KEY_PREFIX)

render_beam_compute_fragment(
    key_prefix=KEY_PREFIX,
    page_key=PAGE_KEY,
    title_prefix="Hermite-Gaussian",
    mat_filename="hermite_gaussian_result.mat",
    get_beam_params=get_hermite_gaussian_params_from_session,
    compute_field=lambda params, start, end, shape: compute_hermite_gaussian(
        params[1], params[2], params[0], params[3], params[4], params[5], start, end, shape
    ),
)
