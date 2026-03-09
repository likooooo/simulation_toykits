"""Fresnel 页面共用的 session 初始化与对 core 的薄封装。"""

from pathlib import Path

import pandas as pd
import streamlit as st

from core import materials as core_materials
from core import refractiveindex as ri


def init_materials_db():
    st.session_state["materials_db"] = {}
    st.session_state["materials_db"]["Vacuum"] = {
        "Shelf ID": "\\",
        "Book ID": "Vacuum",
        "Page ID": "\\",
        "Material Name": "Vacuum",
        "Data Source": "\\",
    }


def init_layer_config():
    st.session_state["layer_config"] = pd.DataFrame(
        [{"Material": "Vacuum", "Thickness (um)": 0.0, "n": 1, "k": 0}]
    )


def ensure_fresnel_session_state():
    """保证 Fresnel 相关页面用到的 session 键都存在（新 session 或直接打开子页时必调）。"""
    if "materials_db" not in st.session_state:
        init_materials_db()
    if "layer_config" not in st.session_state:
        init_layer_config()
    if "wavelength" not in st.session_state:
        st.session_state["wavelength"] = 0.532
    if "degree" not in st.session_state:
        st.session_state["degree"] = 15
    if "film_stack_code" not in st.session_state:
        st.session_state["film_stack_code"] = (
            "Vacuum 0 1 0 SiO2  0.12874 1.4621 1.4254e-5 Ta2O5  0.04396 2.1548  0.00021691 "
            "SiO2 0.27602 1.4621 1.4254e-5 Ta2O5 0.01699 2.1548  0.00021691  "
            "SiO2  0.24735 1.4621 1.4254e-5 fused_silica 0 1.4607 0"
        )


ensure_fresnel_session_state()


def get_nk_at_wavelength(name, wl_um):
    """从 session 材料库与波长取 nk，依赖 core.materials；错误时展示 st.error 并返回 1+0j。"""
    materials_db = st.session_state.get("materials_db", {})
    try:
        return core_materials.get_nk_at_wavelength(
            materials_db, name, wl_um, ri.load_nk
        )
    except Exception:
        st.error(
            f"加载材料 {name} (@ {wl_um} um) 出错.\n"
            "1. 请在 Material Database 中添加材料;\n2. 检查材料波长在范围内;\n"
        )
        return 1.0 + 0.0j


def with_nk_columns(df, wl_um):
    """为层配置表补全 n、k 列，通过 core.materials；get_nk 使用当前 session 材料库。"""
    materials_db = st.session_state.get("materials_db", {})

    def get_nk(name):
        try:
            return core_materials.get_nk_at_wavelength(
                materials_db, name, wl_um, ri.load_nk
            )
        except Exception:
            return 1.0 + 0.0j

    return core_materials.with_nk_columns(df, wl_um, get_nk)


def ensure_beams_session_state():
    """Ensure beams_result_cache exists so each beam page can store/restore results."""
    if "beams_result_cache" not in st.session_state:
        st.session_state["beams_result_cache"] = {}


def page_key_from_file(file_path: str) -> str:
    """Stable page key from script path, e.g. 'pages/beams_caculator/plane-wave.py' -> 'plane-wave'."""
    return Path(file_path).stem

def page_grid_init(key_prefix=""):
    """Render grid inputs, return (x_min, x_max, y_min, y_max, nx, ny)."""
    p = f"{key_prefix}_" if key_prefix else ""
    st.caption("Grid parameters:")
    x_min = st.number_input("start x (µm)", value=-1.1, format="%.2f", key=f"{p}xmin")
    x_max = st.number_input("end   x (µm)", value=1.1, format="%.2f", key=f"{p}xmax")
    y_min = st.number_input("start y (µm)", value=-1.1, format="%.2f", key=f"{p}ymin")
    y_max = st.number_input("end   y (µm)", value=1.1, format="%.2f", key=f"{p}ymax")
    nx = st.number_input("nx", value=100, min_value=2, max_value=256, step=1, key=f"{p}nx")
    ny = st.number_input("ny", value=100, min_value=2, max_value=256, step=1, key=f"{p}ny")
    return x_min, x_max, y_min, y_max, nx, ny


def page_plane_wave_init(key_prefix="pw"):
    """Render plane wave params, return (wavelength, theta_deg, phi_deg)."""
    st.caption("Beam parameters:")
    wavelength = st.number_input("Wavelength (µm)", value=0.11, format="%.4f", key=f"{key_prefix}_wl")
    theta_deg = st.number_input("θ (deg)", min_value = -89.0, max_value = 89.0, value = 10.0, step = 0.1, key=f"{key_prefix}_theta")
    phi_deg = st.number_input("φ (deg)", min_value = -180.0, max_value = 180.0, value = 30.0, step = 0.1, key=f"{key_prefix}_phi")
    return wavelength, theta_deg, phi_deg


def page_quadratic_wave_init(key_prefix="qw"):
    """Render quadratic wave params, return (wavelength, z_ratio)."""
    st.caption("Beam parameters:")
    wavelength = st.number_input("Wavelength (µm)", value=1.1, format="%.4f", key=f"{key_prefix}_wl")
    z_ratio = st.number_input("z ratio (z = z ratio × wavelength)", value=0.25, format="%.2f", key=f"{key_prefix}_zr")
    return wavelength, z_ratio


def page_spherical_wave_init(key_prefix="sw"):
    """Render spherical wave params, return (wavelength, z_ratio)."""
    st.caption("Beam parameters:")
    wavelength = st.number_input("Wavelength (µm)", value=1.1, format="%.4f", key=f"{key_prefix}_wl")
    z_ratio = st.number_input("z ratio (z = z ratio × wavelength)", value=0.25, format="%.2f", key=f"{key_prefix}_zr")
    return wavelength, z_ratio


def page_flat_top_init(key_prefix="ft"):
    """Render flat-top params; return (mode, fraction, r, order, rx, ry, order_x, order_y)."""
    st.caption("Beam parameters:")
    mode = st.radio("Mode", ["Circular", "Rectangular"], key=f"{key_prefix}_mode", horizontal=True)
    fraction = st.number_input(
        "amplitude at radius", value=0.5, min_value=1e-6, max_value=1 - 1e-6, format="%.2f", key=f"{key_prefix}_frac"
    )
    if mode == "Circular":
        r = st.number_input("r (µm)", value=0.8, min_value=0.01, format="%.2f", key=f"{key_prefix}_r")
        order = st.number_input("order", value=5.5, format="%.2f", key=f"{key_prefix}_order")
        rx = ry = order_x = order_y = None
    else:
        rx = st.number_input("rx (µm)", value=0.8, min_value=0.01, format="%.2f", key=f"{key_prefix}_rx")
        ry = st.number_input("ry (µm)", value=0.8, min_value=0.01, format="%.2f", key=f"{key_prefix}_ry")
        order_x = st.number_input("order x", value=5.5, format="%.2f", key=f"{key_prefix}_ox")
        order_y = st.number_input("order y", value=5.5, format="%.2f", key=f"{key_prefix}_oy")
        r = order = None
    return mode, fraction, r, order, rx, ry, order_x, order_y


def page_hermite_gaussian_init(key_prefix="hg"):
    """Render Hermite-Gaussian params, return (wavelength, m, n, z, wx0, wy0)."""
    st.caption("Beam parameters:")
    wavelength = st.number_input("Wavelength (µm)", value=0.5, format="%.4f", key=f"{key_prefix}_wl")
    z_ratio = st.number_input("z ratio (z = z ratio × wavelength)", value=0.25, format="%.2f", key=f"{key_prefix}_zr")
    wx0 = st.number_input("wx0 (µm)", value=1.0, min_value=0.01, format="%.2f", key=f"{key_prefix}_wx0")
    wy0 = st.number_input("wy0 (µm)", value=1.0, min_value=0.01, format="%.2f", key=f"{key_prefix}_wy0")
    m = st.number_input("m", value=3, min_value=0, step=1, key=f"{key_prefix}_m")
    n = st.number_input("n", value=3, min_value=0, step=1, key=f"{key_prefix}_n")
    return wavelength, m, n, z_ratio*wavelength, wx0, wy0


def page_laguerre_gaussian_init(key_prefix="lg"):
    """Render Laguerre-Gaussian params, return (wavelength, p, l, z, w0)."""
    st.caption("Beam parameters:")
    wavelength = st.number_input("Wavelength (µm)", value=0.5, format="%.4f", key=f"{key_prefix}_wl")
    z_ratio = st.number_input("z ratio (z = z ratio × wavelength)", value=0.25, format="%.2f", key=f"{key_prefix}_zr")
    w0 = st.number_input("w0 (µm)", value=1.0, min_value=0.01, format="%.2f", key=f"{key_prefix}_w0")
    p = st.number_input("p", value=3, min_value=0, step=1, key=f"{key_prefix}_p")
    l = st.number_input("l", value=-3, step=1, key=f"{key_prefix}_l")
    return wavelength, p, l, z_ratio*wavelength, w0
