from common import ensure_fresnel_session_state
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

from core import compute_TE_TM_wavelength_angle_figures

ensure_fresnel_session_state()

st.set_page_config(page_title="Angular-spectral map", layout="wide")
st.header("角度-光谱图")

if "coating_films" not in st.session_state:
    st.info("请先在「膜系结构」页完成计算，再查看本页。")
    st.stop()

if "spectral_result_wls" not in st.session_state:
    st.info("请先在「光谱曲线」页点击「计算」，完成波长-反射/透射曲线计算后再查看本页。")
    st.stop()

input_layers = st.session_state["coating_films"]
layer_names = st.session_state["spectral_result_layer_names"]
nk_map = st.session_state["spectral_result_nk_map"]
wls = st.session_state["spectral_result_wls"]
angles_deg = np.linspace(0, 89, 90)

if "angular_fig_te" in st.session_state and "angular_fig_tm" in st.session_state:
    st.subheader("TE 模式")
    st.pyplot(st.session_state["angular_fig_te"])
    st.subheader("TM 模式")
    st.pyplot(st.session_state["angular_fig_tm"])
else:
    try:
        with st.spinner("计算 TE/TM 波长-角度二维图…"):
            fig_te, fig_tm = compute_TE_TM_wavelength_angle_figures(
                input_layers,
                layer_names,
                nk_map,
                wls,
                angles_deg,
            )
        st.session_state["angular_fig_te"] = fig_te
        st.session_state["angular_fig_tm"] = fig_tm
        st.subheader("TE 模式")
        st.pyplot(fig_te)
        st.subheader("TM 模式")
        st.pyplot(fig_tm)
    except Exception as e:
        st.error(f"计算失败: {e}")
