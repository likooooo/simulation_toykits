from common import with_nk_columns, init_layer_config, ensure_fresnel_session_state
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np

from core import (
    parse_formula_v1,
    compute_fresnel_and_filmstack,
)
import simulation_loader
simulation_loader.get_simulation_module()
from assets.simulation import meterial_s

ensure_fresnel_session_state()


def get_available_materials():
    db_materials = list(st.session_state.get("materials_db", {}).keys())
    config_materials = []
    if "layer_config" in st.session_state:
        df = st.session_state["layer_config"]
        if isinstance(df, pd.DataFrame) and "Material" in df.columns:
            config_materials = df["Material"].dropna().unique().tolist()
    combined = db_materials + config_materials
    return list(dict.fromkeys(combined))


st.set_page_config(page_title="Fresnel caculator (build filmstack)", layout="wide")
st.header("膜系结构")

_DOCS_URL = "https://github.com/likooooo/simulation_toykits/blob/main/docs/fresnel_calculator_usage.md"

col_cfg1, col_cfg2 = st.columns(2)
with col_cfg1:
    st.session_state["degree"] = st.slider(
        "入射角度 (Degree)", 0, 89, st.session_state["degree"]
    )
    angle_deg = st.session_state["degree"]
with col_cfg2:
    st.session_state["wavelength"] = st.number_input(
        "参考波长 (um)",
        value=st.session_state["wavelength"],
        format="%.9f",
    )
    target_wl = st.session_state["wavelength"]

with st.expander("🛠️", expanded=True):
    st.caption('**多层膜构建指令** — 📖 [使用说明与语法](%s)' % _DOCS_URL)
    formula_str = st.text_input(
        "多层膜构建指令",
        value=st.session_state["film_stack_code"],
        help="格式示例: Vacuum 0 (SiO2 0.1 Ta2O5 0.01)^5 Vacuum 0",
        label_visibility="collapsed",
    )
    st.session_state["film_stack_code"] = formula_str
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        if st.button("🔄 刷新", width="stretch", key="update film table"):
            if formula_str:
                try:
                    new_data = parse_formula_v1(formula_str)
                    if new_data:
                        st.session_state["layer_config"] = with_nk_columns(
                            pd.DataFrame(new_data), target_wl
                        )
                        st.rerun()
                    else:
                        st.error("未识别到有效的 [材料 厚度] 组合")
                except Exception as e:
                    st.error(f"解析出错: {e}")
    with col_cfg2:
        if st.button("🗑️ 清空", width="stretch", key="clear film table"):
            init_layer_config()
            st.rerun()
    st.session_state["layer_config"] = st.data_editor(
        st.session_state["layer_config"],
        column_config={
            "Material": st.column_config.SelectboxColumn(
                "材料",
                options=get_available_materials(),
                required=True,
            ),
            "Thickness (um)": st.column_config.NumberColumn(
                "厚度 (um)", min_value=0.0, format="%.9f"
            ),
            "n": st.column_config.NumberColumn("n", format="%.9f"),
            "k": st.column_config.NumberColumn("k", format="%.9f"),
        },
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key="film_editor_main",
    )

if st.button("▶️ 计算", width="stretch", key="calculate fresnel coefficients"):
    edited_df = st.session_state["layer_config"]
    if len(edited_df) < 2:
        st.warning("请至少添加两层材料（入射介质和基底）")
    else:
        try:
            with st.spinner("计算 Fresnel 与膜系图…"):
                result = compute_fresnel_and_filmstack(
                    material_factory=lambda: meterial_s(),
                    material_names=edited_df["Material"].tolist(),
                    nk_list=[
                        n + 1j * k
                        for n, k in zip(
                            edited_df["n"].tolist(),
                            edited_df["k"].tolist(),
                        )
                    ],
                    thickness_list=edited_df["Thickness (um)"].tolist(),
                    angle_deg=angle_deg,
                    wl_um=target_wl,
                )
            st.session_state["coating_films"] = result.tmm_layers
            st.session_state["film_result_cache"] = {
                "result": result,
                "wl": target_wl,
                "angle": angle_deg,
            }
            for key in ("spectral_result_wls", "spectral_result_nk_map", "spectral_result_layer_names"):
                st.session_state.pop(key, None)
            st.session_state.pop("spectral_fig_rt", None)
            st.session_state.pop("spectral_fig_nk", None)
            st.session_state.pop("angular_fig_te", None)
            st.session_state.pop("angular_fig_tm", None)
        except Exception as e:
            st.error(f"运行失败: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

if "film_result_cache" in st.session_state:
    cache = st.session_state["film_result_cache"]
    result = cache["result"]
    wl_display = cache["wl"]
    angle_display = cache["angle"]
    st.divider()
    st.subheader(f"📊 仿真结果 (@ {wl_display} μm, {angle_display}°)")
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.write("#### TE 模式")
        c1, c2 = st.columns(2)
        c1.metric("Reflectance (R)", f"{result.R_s:.4f}")
        c2.metric("Transmittance (T)", f"{result.T_s:.4f}")
        with st.expander("fresnel coefficients", expanded=True):
            st.write(f"r: `{result.r_s:.4f}`")
            st.write(f"t: `{result.t_s:.4f}`")
    with res_col2:
        st.write("#### TM 模式")
        c3, c4 = st.columns(2)
        c3.metric("Reflectance (R)", f"{result.R_p:.4f}")
        c4.metric("Transmittance (T)", f"{result.T_p:.4f}")
        with st.expander("fresnel coefficients", expanded=True):
            st.write(f"r: `{result.r_p:.4f}`")
            st.write(f"t: `{result.t_p:.4f}`")
    st.pyplot(result.filmstack_fig)
