from pages.TMM.films import get_nk_at_wavelength
import matplotlib.pyplot as plt
from assets.simulation import *
import streamlit as st
import numpy as np

st.set_page_config(page_title="Simulation toykits (TMM)", layout="wide")
st.header("光谱曲线")

def compute_RT(input_layers, th0, wl):
    dir = TMM_propagate_direction(input_layers, th0)

    tmm_s = TMM_interface_transfer_matrix_with_thickness_s(input_layers, dir, wl)
    tmm_p = TMM_interface_transfer_matrix_with_thickness_p(input_layers, dir, wl)

    R_s, T_s = TMM_get_r_t_power_from_tmm_s(
        tmm_s[-1], input_layers[0].nk, dir[0],
        input_layers[-1].nk, dir[-1]
    )
    R_p, T_p = TMM_get_r_t_power_from_tmm_p(
        tmm_p[-1], input_layers[0].nk, dir[0],
        input_layers[-1].nk, dir[-1]
    )

    return R_s, T_s, R_p, T_p

if 'coating_films' in st.session_state: 
    input_layers = st.session_state['coating_films']

    angles = np.linspace(0, 89, 90)
    Rs, Rp = [], []
    Ts, Tp = [], []
    wl0 = st.session_state['wavelength']
    for ang in angles:
        R_s, T_s, R_p, T_p = compute_RT(
            input_layers,
            ang * np.pi / 180,
            wl0
        )
        Rs.append(R_s)
        Rp.append(R_p)
        Ts.append(T_s)
        Tp.append(T_p)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(angles, Rs, label="TE", color='blue')
    ax1.plot(angles, Rp, label="TM", color='red')
    ax1.set_xlabel("Angle (deg)")
    ax1.set_ylabel("Reflectance")
    ax1.set_title(f"Reflectance (@{wl0}um)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.plot(angles, Ts, label="TE", color='blue')
    ax2.plot(angles, Tp, label="TM", color='red')
    ax2.set_xlabel("Angle (deg)")
    ax2.set_ylabel("Transmittance") 
    ax2.set_title(f"Transmittance (@{wl0}um)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()
    st.pyplot(fig)


    col_wl_min, col_wl_max = st.columns(2)
    with col_wl_min:
        wl_min = st.number_input(
            "起始波长 (um)", 
            min_value=0.1, 
            max_value=10.0, 
            value=0.400, 
            format="%.4f",
            step=0.01
        )

    with col_wl_max:
        wl_max = st.number_input(
            "截止波长 (um)", 
            min_value=0.1, 
            max_value=10.0, 
            value=0.800, 
            format="%.4f",
            step=0.01
        )

    if wl_min >= wl_max: st.error("错误：起始波长必须小于截止波长")

    if st.button("▶️ 计算", width='stretch'):
        wls = np.linspace(wl_min, wl_max, 100)
        Rs, Rp = [], []
        Ts, Tp = [], []
        angle_deg = st.session_state['degree'] 
        nk_map = {}
        layer_names = st.session_state['layer_config']['Material']
        for material_name in layer_names:
            if material_name in nk_map: continue
            nk_map[material_name] = [get_nk_at_wavelength(material_name, w) for w in wls]
        for i in range(len(wls)):
            wl = wls[i]
            # update nk
            for name, layer in zip(layer_names, input_layers):
                layer.nk = nk_map[name][i]
            R_s, T_s, R_p, T_p = compute_RT(
                input_layers,
                angle_deg * np.pi / 180,
                wl
            )
            Rs.append(R_s)
            Rp.append(R_p)
            Ts.append(T_s)
            Tp.append(T_p)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(wls, Rs, label="TE", color='blue')
        ax1.plot(wls, Rp, label="TM", color='red')
        ax1.set_xlabel("Wavelength (um)")
        ax1.set_ylabel("Reflectance")
        ax1.set_title(f"Reflectance ({angle_deg}°)")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)

        ax2.plot(wls, Ts, label="TE", color='blue')
        ax2.plot(wls, Tp, label="TM", color='red')
        ax2.set_xlabel("Wavelength (um)")
        ax2.set_ylabel("Transmittance") 
        ax2.set_title(f"Transmittance ({angle_deg}°)")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()
        st.pyplot(fig)

        fig, ax_list = plt.subplots(1, len(nk_map.keys()), figsize=(12, 5))
        for i, (name, nk_list) in enumerate(nk_map.items()):
            ax = ax_list[i]
            n = [np.real(nk) for nk in nk_list]
            k = [np.imag(nk) for nk in nk_list]

            ax.plot(wls, n, label="n", color='blue')
            ax.plot(wls, k, label="k", color='red')
            ax.set_xlabel("Wavelength (um)")
            ax.set_ylabel("n,k")
            ax.set_title(name)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()
        st.pyplot(fig)
