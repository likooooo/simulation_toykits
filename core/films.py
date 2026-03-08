"""
膜系单次计算：Fresnel 系数 + 膜系结构图。一次调用返回所有结果，无 st 依赖。
"""

from typing import List, Any, Callable, NamedTuple
import numpy as np

from core.fresnel import build_tmm_layers, compute_RT, get_r_t
from core.filmstack_viz import (
    build_layers_for_visualization,
    calculate_angles,
    nk_to_color,
    plot_periodic_structure,
)


class FresnelFilmstackResult(NamedTuple):
    """Fresnel 计算 + 膜系图的一次性结果。"""

    tmm_layers: List[Any]
    R_s: float
    T_s: float
    R_p: float
    T_p: float
    r_s: complex
    t_s: complex
    r_p: complex
    t_p: complex
    filmstack_fig: Any  # matplotlib.Figure


def compute_fresnel_and_filmstack(
    material_factory: Callable[[], Any],
    material_names: List[str],
    nk_list: List[complex],
    thickness_list: List[float],
    angle_deg: float,
    wl_um: float,
) -> FresnelFilmstackResult:
    """
    计算单波长单角度下的 Fresnel R/T、r/t，并绘制膜系结构图。无 st 依赖。

    :param material_factory: 无参调用返回 TMM 层对象（如 lambda: meterial_s()）
    :param material_names: 每层材料名
    :param nk_list: 每层复折射率
    :param thickness_list: 每层厚度 (μm)
    :param angle_deg: 入射角 (度)
    :param wl_um: 波长 (μm)
    :return: FresnelFilmstackResult，含 tmm_layers、R_s/T_s/R_p/T_p、r_s/t_s/r_p/t_p、filmstack_fig
    """
    tmm_layers = build_tmm_layers(
        material_factory, nk_list, thickness_list
    )
    th_0_rad = np.deg2rad(angle_deg)
    R_s, T_s, R_p, T_p = compute_RT(tmm_layers, th_0_rad, wl_um)
    r_s, t_s, r_p, t_p = get_r_t(tmm_layers, th_0_rad, wl_um)

    layers = build_layers_for_visualization(
        material_names, nk_list, thickness_list
    )
    angles = calculate_angles(layers, th_0_rad)
    all_n = [l.nk.real for l in layers]
    all_k = [l.nk.imag for l in layers]
    nmin, nmax = min(all_n), max(all_n)
    k_max = max(all_k) if max(all_k) > 0 else 1.0
    color_map = {}
    for layer in layers:
        if layer.name in color_map:
            continue
        color_map[layer.name] = nk_to_color(
            layer.nk.real, layer.nk.imag, nmin, nmax, k_max
        )
    filmstack_fig = plot_periodic_structure(
        layers,
        angles,
        color_map,
        angle_deg=angle_deg,
        title=f"Filmstack Visualization (@{wl_um} μm)",
        visual_width=-1,
        inf_display_height=max(np.mean(thickness_list), wl_um),
    )

    return FresnelFilmstackResult(
        tmm_layers=tmm_layers,
        R_s=R_s,
        T_s=T_s,
        R_p=R_p,
        T_p=T_p,
        r_s=r_s,
        t_s=t_s,
        r_p=r_p,
        t_p=t_p,
        filmstack_fig=filmstack_fig,
    )
