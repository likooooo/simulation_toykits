from common import with_nk_columns, init_layer_config
from filmstack_visualization import *
from assets.simulation import *
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import re

meterial = meterial_s

def get_available_materials():
    db_materials = list(st.session_state.get('materials_db', {}).keys())
    config_materials = []
    if 'layer_config' in st.session_state:
        df = st.session_state['layer_config']
        if isinstance(df, pd.DataFrame) and 'Material' in df.columns:
            config_materials = df['Material'].dropna().unique().tolist()
    
    combined = db_materials + config_materials
    return list(dict.fromkeys(combined))

def parse_formula(formula: str):
    """
    支持格式: (H L)^10, (H 0.1 L 0.2)^5, SiO2 0.1 TiO2 0.2
    """
    while '(' in formula:
        match = re.search(r'\(([^()]+)\)\^(\d+)', formula)
        if not match: break
        content, times = match.group(1), int(match.group(2))
        formula = formula.replace(match.group(0), (content + " ") * times)
    
    pairs = re.findall(r'(\S+)\s+([\d.]+)', formula)
    return [{"Material": m, "Thickness (um)": float(t)} for m, t in pairs]

def parse_formula_v1(formula: str):
    """
    支持语法：
    - Material Thickness
    - Material Thickness n k

    支持括号周期：
    (SiO2 0.1 1.5 0.001 Ta2O5 0.01)^5
    """

    while '(' in formula:
        match = re.search(r'\(([^()]+)\)\^(\d+)', formula)
        if not match:
            break
        content, times = match.group(1), int(match.group(2))
        formula = formula.replace(match.group(0), (content + " ") * times)

    tokens = formula.split()
    layers = []
    i = 0

    while i < len(tokens):
        material = tokens[i]
        thickness = float(tokens[i + 1])

        n_override = None
        k_override = None

        # 判断是否存在 n k
        if i + 3 < len(tokens):
            try:
                n_test = float(tokens[i + 2])
                k_test = float(tokens[i + 3])
                n_override = n_test
                k_override = k_test
                i += 4
            except ValueError:
                i += 2
        else:
            i += 2
        # data = {
        #     "Material": material,
        #     "Thickness (um)": thickness,
        # }
        # if n_override is not None and k_override is not None:
        #     data["n"] = n_override
        #     data["k"] = k_override
        # layers.append(data)

        layers.append({
            "Material": material,
            "Thickness (um)": thickness,
            "n": n_override,
            "k": k_override
        })
    return layers

st.set_page_config(page_title="Fresnel caculator (build filmstack)", layout="wide")
st.header("膜系结构")

col_cfg1, col_cfg2 = st.columns(2)
with col_cfg1:
    st.session_state['degree'] = st.slider("入射角度 (Degree)", 0, 89, st.session_state['degree'])
    angle_deg = st.session_state['degree']
with col_cfg2:
    st.session_state['wavelength'] = st.number_input("参考波长 (um)", value=st.session_state['wavelength'], format="%.9f")
    target_wl = st.session_state['wavelength']

with st.expander("🛠️", expanded=True):
    formula_str = st.text_input(
        "公式语法", 
        value=st.session_state['film_stack_code'], 
        help="格式示例: Vacuum 0 (SiO2 0.1 Ta2O5 0.01)^5 Vacuum 0"
    )
    st.session_state['film_stack_code'] = formula_str
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        if st.button("🔄 刷新", width='stretch', key="update film table"):
            if formula_str:
                try:
                    new_data = parse_formula_v1(formula_str)
                    if new_data:
                        st.session_state['layer_config'] = with_nk_columns(pd.DataFrame(new_data), target_wl)
                        st.rerun()
                    else:
                        st.error("未识别到有效的 [材料 厚度] 组合")
                except Exception as e:
                    st.error(f"解析出错: {e}")
    with col_cfg2:
        if st.button("🗑️ 清空", width='stretch', key="clear film table"):
            init_layer_config()
            st.rerun()
    st.session_state['layer_config'] = st.data_editor(
        st.session_state['layer_config'],
        column_config={
            "Material": st.column_config.SelectboxColumn(
                "材料",
                options=get_available_materials(),
                required=True
            ),
            "Thickness (um)": st.column_config.NumberColumn(
                "厚度 (um)",
                min_value=0.0, format="%.9f"
            ),
            "n": st.column_config.NumberColumn(
                "n", format="%.9f"
            ),
            "k": st.column_config.NumberColumn(
                "k", format="%.9f"
            )
        },
        num_rows="dynamic",
        hide_index=True,
        width='stretch',
        key="film_editor_main"
    )

if st.button("▶️ 计算", width='stretch', key="calculate fresnel coefficients"):
    edited_df = st.session_state['layer_config']  
    if len(edited_df) < 2:
        st.warning("请至少添加两层材料（入射介质和基底）")
    else:
        try:
            meterial_film_name_list = edited_df["Material"].tolist()
            thickness_list = edited_df["Thickness (um)"].tolist()
            def get_nk(i):
                i = len(st.session_state['layer_config']['n']) - 1 if -1==i else i
                return st.session_state['layer_config']['n'][i] + 1j* st.session_state['layer_config']['k'][i]
            upper = meterial()
            upper.nk = get_nk(0)
            substrate = meterial()
            substrate.nk = get_nk(-1)

            def make_film(nk, depth):
                m = meterial()
                m.nk = nk
                m.depth = depth
                return m
            input = [upper]
            for i in range(1, len(thickness_list)-1):
                nk_val = get_nk(i)
                input.append(make_film(nk_val, thickness_list[i]))
            input.append(substrate)
            st.session_state['coating_films'] = input
            wl = target_wl
            th_0 = angle_deg * (np.pi / 180)
            dir = TMM_propagate_direction(input, th_0)

            tmm_s = TMM_interface_transfer_matrix_with_thickness_s(input, dir, wl)
            tmm_p = TMM_interface_transfer_matrix_with_thickness_p(input, dir, wl)
            r_s, t_s = TMM_get_r_t_from_tmm(tmm_s[-1])
            r_p, t_p = TMM_get_r_t_from_tmm(tmm_p[-1])
            R_s, T_s = TMM_get_r_t_power_from_tmm_s(tmm_s[-1], input[0].nk, dir[0], input[-1].nk, dir[-1])
            R_p, T_p = TMM_get_r_t_power_from_tmm_p(tmm_p[-1], input[0].nk, dir[0], input[-1].nk, dir[-1])

            st.divider()
            st.subheader(f"📊 仿真结果 (@ {target_wl} μm, {angle_deg}°)")
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.write("#### TE Mode")
                c1, c2 = st.columns(2)
                c1.metric("Reflectance (R)", f"{R_s:.4f}")
                c2.metric("Transmittance (T)", f"{T_s:.4f}")
                with st.expander("fresnel coefficients", expanded=True):
                    st.write(f"r: `{r_s:.4f}`")
                    st.write(f"t: `{t_s:.4f}`")

            with res_col2:
                st.write("#### TM Mode")
                c3, c4 = st.columns(2)
                c3.metric("Reflectance (R)", f"{R_p:.4f}")
                c4.metric("Transmittance (T)", f"{T_p:.4f}")
                with st.expander("fresnel coefficients", expanded=True):
                    st.write(f"r: `{r_p:.4f}`")
                    st.write(f"t: `{t_p:.4f}`")
            
            layers = []
            for i, name in enumerate(meterial_film_name_list):
                nk = get_nk(i)
                mat = MockMaterial(name, nk)
                d = float('inf')  if i == 0 or i == len(meterial_film_name_list) - 1 else thickness_list[i]
                layers.append(MockFilm(mat, d))

            angles = calculate_angles(layers, th_0)
            all_n = [l.nk.real for l in layers]
            all_k = [l.nk.imag for l in layers]
            nmin, nmax = min(all_n), max(all_n)
            k_max = max(all_k) if max(all_k) > 0 else 1
            
            color_map = {}
            for l in layers:
                if l.name  in color_map: continue
                color_map[l.name] = nk_to_color(l.nk.real, l.nk.imag, nmin, nmax, k_max)

            plot_periodic_structure(
                layers, angles, color_map, 
                angle_deg=angle_deg, 
                title=f"Filmstack Visualization (@{target_wl}um)",
                visual_width=-1, inf_display_height=max(np.mean(thickness_list), target_wl)
            )
            st.pyplot(plt.gcf())

        except Exception as e:
            st.error(f"运行失败: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
