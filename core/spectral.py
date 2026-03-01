"""
光谱曲线计算与绘图：角度- R/T、波长- R/T、材料 nk 曲线。无 st 依赖，返回 matplotlib 或 plotly 对象。
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt


def _get_compute_RT():
    """延迟导入，避免 CI 下加载 core.fresnel（依赖 assets.simulation）导致失败。"""
    from core.fresnel import compute_RT
    return compute_RT


def compute_angle_vs_RT_figures(
    layers: List[Any],
    wl_um: float,
    angles_deg: np.ndarray,
) -> List[plt.Figure]:
    """
    固定波长下计算 R/T 随角度的变化，返回一张图（Reflectance + Transmittance）。

    :param layers: TMM 层列表（build_tmm_layers 的返回值）
    :param wl_um: 波长 (μm)
    :param angles_deg: 角度数组 (度)
    :return: [fig]，fig 含两个子图 R、T vs angle
    """
    Rs, Rp = [], []
    Ts, Tp = [], []
    compute_RT = _get_compute_RT()
    for ang in angles_deg:
        th_rad = np.deg2rad(ang)
        R_s, T_s, R_p, T_p = compute_RT(layers, th_rad, wl_um)
        Rs.append(R_s)
        Rp.append(R_p)
        Ts.append(T_s)
        Tp.append(T_p)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(angles_deg, Rs, label="TE", color="blue")
    ax1.plot(angles_deg, Rp, label="TM", color="red")
    ax1.set_xlabel("Angle (deg)")
    ax1.set_ylabel("Reflectance")
    ax1.set_title(f"Reflectance (@{wl_um} μm)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    ax2.plot(angles_deg, Ts, label="TE", color="blue")
    ax2.plot(angles_deg, Tp, label="TM", color="red")
    ax2.set_xlabel("Angle (deg)")
    ax2.set_ylabel("Transmittance")
    ax2.set_title(f"Transmittance (@{wl_um} μm)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()
    return [fig]


def compute_wavelength_vs_RT_figures(
    layers: List[Any],
    layer_names: List[str],
    nk_map: Dict[str, List[complex]],
    wls: np.ndarray,
    angle_deg: float,
) -> Tuple[plt.Figure, plt.Figure]:
    """
    计算 R/T 随波长的变化，并绘制材料 n-k 随波长曲线。会就地修改 layers 的 .nk。

    :param layers: TMM 层列表（可变，每步更新 .nk）
    :param layer_names: 每层材料名，与 layers 同序
    :param nk_map: 材料名 -> 该材料在各波长下的 nk 列表，长度 = len(wls)
    :param wls: 波长数组 (μm)
    :param angle_deg: 入射角 (度)
    :return: (fig_rt, fig_nk)，分别为 R/T vs 波长 与 各材料 n,k vs 波长
    """
    compute_RT = _get_compute_RT()
    angle_rad = np.deg2rad(angle_deg)
    Rs, Rp = [], []
    Ts, Tp = [], []
    for i in range(len(wls)):
        for name, layer in zip(layer_names, layers):
            layer.nk = nk_map[name][i]
        R_s, T_s, R_p, T_p = compute_RT(layers, angle_rad, wls[i])
        Rs.append(R_s)
        Rp.append(R_p)
        Ts.append(T_s)
        Tp.append(T_p)

    fig_rt, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(wls, Rs, label="TE", color="blue")
    ax1.plot(wls, Rp, label="TM", color="red")
    ax1.set_xlabel("Wavelength (μm)")
    ax1.set_ylabel("Reflectance")
    ax1.set_title(f"Reflectance ({angle_deg}°)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax2.plot(wls, Ts, label="TE", color="blue")
    ax2.plot(wls, Tp, label="TM", color="red")
    ax2.set_xlabel("Wavelength (μm)")
    ax2.set_ylabel("Transmittance")
    ax2.set_title(f"Transmittance ({angle_deg}°)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)
    fig_rt.tight_layout()

    n_materials = len(nk_map)
    fig_nk, ax_list = plt.subplots(
        1, max(1, n_materials), figsize=(4 * max(1, n_materials), 5)
    )
    if n_materials == 1:
        ax_list = [ax_list]
    for i, (name, nk_list) in enumerate(nk_map.items()):
        ax = ax_list[i]
        n_vals = [np.real(nk) for nk in nk_list]
        k_vals = [np.imag(nk) for nk in nk_list]
        ax.plot(wls, n_vals, label="n", color="blue")
        ax.plot(wls, k_vals, label="k", color="red")
        ax.set_xlabel("Wavelength (μm)")
        ax.set_ylabel("n, k")
        ax.set_title(name)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)
    fig_nk.tight_layout()
    return fig_rt, fig_nk


def build_nk_map_for_wavelengths(
    layer_names: List[str],
    n_col: List[float],
    k_col: List[float],
    wls: np.ndarray,
    materials_db: Dict[str, Any],
    get_nk_at_wavelength_fn: Any,
) -> Tuple[Dict[str, List[complex]], List[str]]:
    """
    根据层配置与波长列表构建 nk_map，并返回不在材料库中的材料名列表（用于 UI 提示）。

    :param layer_names: 层材料名
    :param n_col: 层 n 列
    :param k_col: 层 k 列
    :param wls: 波长数组
    :param materials_db: 材料库 dict
    :param get_nk_at_wavelength_fn: (name, wl_um) -> complex
    :return: (nk_map, materials_not_in_db)
    """
    nk_map = {}
    materials_not_in_db = []
    for material_name, n, k in zip(layer_names, n_col, k_col):
        if material_name in nk_map:
            continue
        if material_name not in materials_db:
            nk_map[material_name] = [n + 1j * k] * len(wls)
            materials_not_in_db.append(material_name)
        else:
            nk_map[material_name] = [
                get_nk_at_wavelength_fn(material_name, w) for w in wls
            ]
    return nk_map, materials_not_in_db
