"""
Streamlit 页面共用的 session 状态初始化及对 core 的薄封装（仅负责从 session 取数、展示错误）。
"""

from core import refractiveindex as ri
import streamlit as st
import pandas as pd

from core import materials as core_materials


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
