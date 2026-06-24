import streamlit as st
from common import (
    ensure_beams_session_state,
    page_key_from_file,
    page_grid_init,
    page_flat_top_init,
    get_flat_top_params_from_session,
    render_beam_compute_fragment,
)
from core import compute_flat_top_rectangular, compute_flat_top_circular

ensure_beams_session_state()
PAGE_KEY = page_key_from_file(__file__)
KEY_PREFIX = "ft"


def _compute_flat_top(params, start_xy, end_xy, shape_xy):
    mode, fraction, r, order, rx, ry, order_x, order_y = params
    if mode == "Circular":
        return compute_flat_top_circular(r, fraction, order, start_xy, end_xy, shape_xy)
    return compute_flat_top_rectangular(
        rx, ry, fraction, order_x, order_y, start_xy, end_xy, shape_xy
    )


st.set_page_config(page_title="Flat-Top Beam", layout="wide")
st.header("Flat-Top Beam")

st.subheader("Parameters")
col_grid, col_beam = st.columns(2)
with col_grid:
    page_grid_init(KEY_PREFIX)
with col_beam:
    page_flat_top_init(KEY_PREFIX)

render_beam_compute_fragment(
    key_prefix=KEY_PREFIX,
    page_key=PAGE_KEY,
    title_prefix="Flat-Top Beam",
    mat_filename="flat_top_beam_result.mat",
    get_beam_params=get_flat_top_params_from_session,
    compute_field=_compute_flat_top,
)
