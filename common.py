import refractiveindex as ri
import streamlit as st
import pandas as pd
import numpy as np

def init_materials_db():
    st.session_state['materials_db'] = {}
    st.session_state['materials_db']["Vacuum"] = {
        "Shelf ID": "\\",
        "Book ID": "Vacuum",
        "Page ID": "\\",
        "Material Name": "Vacuum",
        "Data Source": "\\"
    }

if 'materials_db' not in st.session_state: init_materials_db()

def init_layer_config():
     st.session_state['layer_config'] = pd.DataFrame( [{"Material": "Vacuum", "Thickness (um)": 0.0,"n":1,"k":0}])

if 'layer_config' not in st.session_state: init_layer_config()
if 'wavelength' not in st.session_state : st.session_state['wavelength'] = 0.532
if 'degree' not in st.session_state : st.session_state['degree'] = 15
if 'film_stack_code' not in st.session_state : st.session_state['film_stack_code'] = 'Vacuum 0 1 0 SiO2  0.12874 1.4621 1.4254e-5 Ta2O5  0.04396 2.1548  0.00021691 SiO2 0.27602 1.4621 1.4254e-5 Ta2O5 0.01699 2.1548  0.00021691  SiO2  0.24735 1.4621 1.4254e-5 fused_silica 0 1.4607 0'


def get_nk_at_wavelength(name, wl_um):
    if name == "Vacuum": return 1.0 + 0.0j
    try:
        row = st.session_state['materials_db'][name]
        wls, ns, ks = ri.load_nk(row["Shelf ID"], row["Book ID"], row["Page ID"])
        n_val = np.interp(wl_um, wls, ns)
        k_val = np.interp(wl_um, wls, ks)
        return n_val + 1j * k_val
    except Exception as e:
        st.error(f"加载材料 {name} (@ {wl_um} um) 出错.\n1. 请在 Material Database 中添加材料;\n2. 检查材料波长在范围内;\n")
        return 1.0 + 0.0j

def with_nk_columns(df, wl_um):
    if df.empty:
        return df
    df_show = df.copy()
    n_list, k_list = [], []
    for _, row in df.iterrows():
        if row["n"] is not None and row["k"] is not None :
            n_list.append(float(row["n"]))
            k_list.append(float(row["k"]))
            continue
        nk = get_nk_at_wavelength(row["Material"], wl_um)
        n_list.append(np.real(nk))
        k_list.append(np.imag(nk))

    df_show["n"] = n_list
    df_show["k"] = k_list
    return df_show