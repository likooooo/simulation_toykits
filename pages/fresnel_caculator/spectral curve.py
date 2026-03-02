from common import get_nk_at_wavelength, ensure_fresnel_session_state
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

from core import (
    compute_angle_vs_RT_figures,
    compute_wavelength_vs_RT_figures,
    build_nk_map_for_wavelengths,
)

ensure_fresnel_session_state()

st.set_page_config(page_title="Fresnel caculator (result)", layout="wide")
st.header("光谱曲线")

if "coating_films" not in st.session_state:
    st.info("请先在「膜系结构」页完成计算，再查看光谱曲线。")
    st.stop()

input_layers = st.session_state["coating_films"]
wl0 = st.session_state["wavelength"]

# 固定波长下的角度 vs R/T
with st.spinner("计算角度-反射/透射曲线…"):
    angle_figs = compute_angle_vs_RT_figures(
        input_layers, wl0, np.linspace(0, 89, 90)
    )
for fig in angle_figs:
    st.pyplot(fig)
    plt.close(fig)

col_wl_min, col_wl_max = st.columns(2)
with col_wl_min:
    wl_min = st.number_input(
        "起始波长 (um)",
        min_value=0.1,
        max_value=10.0,
        value=0.400,
        format="%.4f",
        step=0.01,
    )
with col_wl_max:
    wl_max = st.number_input(
        "截止波长 (um)",
        min_value=0.1,
        max_value=10.0,
        value=0.800,
        format="%.4f",
        step=0.01,
    )

if wl_min >= wl_max:
    st.error("错误：起始波长必须小于截止波长")
else:
    if st.button("▶️ 计算", width="stretch", key="calculate wavelength vs RT"):
        wls = np.linspace(wl_min, wl_max, 100)
        angle_deg = st.session_state["degree"]
        layer_names = st.session_state["layer_config"]["Material"].tolist()
        n_col = st.session_state["layer_config"]["n"].tolist()
        k_col = st.session_state["layer_config"]["k"].tolist()
        materials_db = st.session_state.get("materials_db", {})

        nk_map, materials_not_in_db = build_nk_map_for_wavelengths(
            layer_names,
            n_col,
            k_col,
            wls,
            materials_db,
            get_nk_at_wavelength,
        )
        for name in materials_not_in_db:
            st.warning(
                f"材料数据库中不存在 {name}, 该材料 nk 不会随着波长变化."
            )

        st.session_state["spectral_result_wls"] = wls
        st.session_state["spectral_result_nk_map"] = nk_map
        st.session_state["spectral_result_layer_names"] = layer_names
        st.session_state.pop("angular_fig_te", None)
        st.session_state.pop("angular_fig_tm", None)

        try:
            with st.spinner("计算波长-反射/透射曲线…"):
                fig_rt, fig_nk = compute_wavelength_vs_RT_figures(
                    input_layers,
                    layer_names,
                    nk_map,
                    wls,
                    angle_deg,
                )
            st.session_state["spectral_fig_rt"] = fig_rt
            st.session_state["spectral_fig_nk"] = fig_nk
        except Exception as e:
            st.error(f"计算失败: {e}")

if "spectral_fig_rt" in st.session_state and "spectral_fig_nk" in st.session_state:
    st.pyplot(st.session_state["spectral_fig_rt"])
    st.pyplot(st.session_state["spectral_fig_nk"])
