"""Diffraction angle calculator: calls ``test_diffraction`` CLI (next to simulation.so)."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from common import ensure_fresnel_session_state, get_nk_at_wavelength, pyplot_fixed_width
from core.diffraction_cli import (
    build_polar_diffraction_figure,
    diffraction_binary_path,
    pair_order_angle,
    run_diffraction,
)

ensure_fresnel_session_state()

st.set_page_config(page_title="Fresnel caculator (diffraction angle)", layout="wide")
st.header("衍射角计算器")


def get_available_materials():
    db_materials = list(st.session_state.get("materials_db", {}).keys())
    config_materials = []
    if "layer_config" in st.session_state:
        df = st.session_state["layer_config"]
        if hasattr(df, "columns") and "Material" in df.columns:
            config_materials = df["Material"].dropna().unique().tolist()
    combined = db_materials + config_materials
    return list(dict.fromkeys(combined))


exe_path = diffraction_binary_path()
if not exe_path.is_file():
    st.error(
        f"未找到 `test_diffraction` 可执行文件：{exe_path}\n"
        "请编译 simulation 的 `test_diffraction` 目标，并将其复制到 `assets/lib/`（与 `simulation.so` 同目录）。"
    )
    st.stop()

col_a, col_b = st.columns(2)
with col_a:
    mats = get_available_materials()
    m1 = st.selectbox(
        "入射材料",
        options=mats,
        index=mats.index("Vacuum") if "Vacuum" in mats else 0,
        key="diffraction_mat1",
        help="在 Material Database 中添加更多材料数据",
    )
    m2 = st.selectbox(
        "出射材料",
        options=mats,
        index=min(1, len(mats) - 1),
        key="diffraction_mat2",
    )
with col_b:
    L_um = st.number_input(
        "光栅周期 Grating period (µm)",
        min_value=1e-9,
        value=float(st.session_state.get("diffraction_L_um", 5.0)),
        format="%.6f",
    )
    st.session_state["diffraction_L_um"] = L_um
    wl_um = st.number_input(
        "波长 Wavelength (µm)",
        min_value=1e-9,
        value=float(st.session_state.get("wavelength", 0.532)),
        format="%.3f",
    )
    incident_deg = st.slider(
        "入射角 Incident angle (°)",
        min_value=-89.0,
        max_value=89.0,
        value=float(st.session_state.get("diffraction_incident_deg", 0.0)),
        step=0.1,
        key="diffraction_incident_deg",
    )


nk1 = get_nk_at_wavelength(m1, wl_um)
nk2 = get_nk_at_wavelength(m2, wl_um)
n_from = float(np.real(nk1))
n_to = float(np.real(nk2))

with col_a:
    max_display_order = st.number_input(
        "Maximum Shown Order",
        value=int(st.session_state.get("diffraction_max_display_order", 3)),
        min_value=0,
        step=1,
        help="一个朴实无华的低通滤波器, 例如设为 3 时，透射与反射均只显示 m ∈ [−3, +3] 且计算结果中存在的级次。",
    )
st.session_state["diffraction_max_display_order"] = max_display_order

st.markdown(
    f"**n (@ {wl_um} µm)** from `{m1}`(n={n_from:.3f})"
    f" to `{m2}`(n={n_to:.6f})"
)

errors = []
if L_um <= 0:
    errors.append("光栅周期必须 > 0")
if wl_um <= 0:
    errors.append("波长必须 > 0")
if not (-90 < incident_deg < 90):
    errors.append("入射角须在 (-90°, 90°) 内")
if n_from <= 0 or n_to <= 0:
    errors.append("两侧材料在该波长下的折射率实部必须 > 0（请检查材料库与波长范围）")

for e in errors:
    st.error(e)

data = None
if not errors:
    try:
        with st.spinner("运行 test_diffraction…"):
            data = run_diffraction(L_um, wl_um, n_from, n_to, incident_deg)
    except Exception as e:
        st.error(f"运行失败: {e}")

if data is not None:
    tj = data["transmitted"]
    rj = data["reflected"]

    t_pairs = pair_order_angle(tj)
    r_pairs = pair_order_angle(rj)

    n_show = int(max_display_order)
    t_show = [(m, a) for m, a in t_pairs if abs(m) <= n_show]
    r_show = [(m, a) for m, a in r_pairs if abs(m) <= n_show]

    st.divider()

    tc1, tc2 = st.columns([1,4])
    with tc1:
        st.subheader("结果")
        st.markdown(
            f"- **入射角**: {incident_deg:g}°\n"
            f"- **透射** 级次范围: **{tj['min']} ~ {tj['max']}**；"
            f"显示 **|m| ≤ {n_show}**（共 {len(t_show)} 条）\n"
            f"- **反射** 级次范围: **{rj['min']} ~ {rj['max']}**；"
            f"显示 **|m| ≤ {n_show}**（共 {len(r_show)} 条）"
        )
        def format_order_row(pairs):
            parts = []
            for m, ang in pairs:
                sign = "+" if m > 0 else ""
                parts.append(f"**{sign}{m}**: {ang:.6g}°")
            return "  \n".join(parts) if parts else "（无）"
        st.markdown("###### Transmitted Orders")
        st.markdown(format_order_row(t_show))
        st.markdown("###### Reflected Orders")
        st.markdown(format_order_row(r_show))
    with tc2:
        fig = build_polar_diffraction_figure(incident_deg, t_show, r_show)
        pyplot_fixed_width(fig, width=700)
        plt.close(fig)
