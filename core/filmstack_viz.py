"""
膜系可视化：角度计算、周期性结构绘图、颜色映射。不依赖全局变量。
"""

import colorsys
from typing import List, Dict, Tuple, Optional, Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class MockMaterial:
    def __init__(self, name: str, nk: complex):
        self.name = name
        self.nk = nk


class MockFilm:
    def __init__(self, material: MockMaterial, thickness: float):
        self.material = material
        self.d = thickness
        self.name = material.name
        self.nk = material.nk


def calculate_angles(
    layer_list: List[MockFilm],
    initial_theta: float,
) -> List[complex]:
    """Snell 定律计算各层折射角。"""
    angles = []
    n0 = layer_list[0].nk.real
    sne_const = n0 * np.sin(initial_theta)
    for layer in layer_list:
        sin_th = sne_const / layer.nk
        theta = np.arcsin(sin_th)
        angles.append(theta)
    return angles


def nk_to_color(
    n: float,
    k: float,
    n_min: float = 1.0,
    n_max: float = 5.0,
    k_max: float = 3.0,
) -> Tuple[float, float, float]:
    """根据 n、k 返回 (r, g, b) 元组用于绘图。"""
    n_norm = (n - n_min) / (n_max - n_min + 1e-6)
    H = (n_norm * 2.5) % 1.0
    k_norm = np.clip(k / k_max, 0.0, 1.0)
    S = 0.5 + 0.5 * (1 - k_norm)
    V = 0.4 + 0.5 * k_norm
    r, g, b = colorsys.hsv_to_rgb(H, S, V)
    return (r, g, b)


def plot_periodic_structure(
    layers: List[MockFilm],
    angles: List[complex],
    color_map: Dict[str, Tuple[float, float, float]],
    angle_deg: float = -1,
    title: str = "Multilayer Ray Tracing",
    visual_width: float = -1,
    inf_display_height: float = 100,
) -> plt.Figure:
    """绘制周期性膜系结构图（射线追踪），返回 Figure 供调用方显示或保存。"""
    fig, ax = plt.subplots(figsize=(12, 10))
    films = layers[1:-1]
    total_film_thickness = sum(f.d for f in films)

    layer_coords = []
    current_top = -inf_display_height
    z_zero_level = 0

    top = current_top
    bottom = z_zero_level
    layer_coords.append((top, bottom))
    current_top = bottom

    for layer in layers[1:]:
        height = inf_display_height if layer.d == float("inf") else layer.d
        top = current_top
        bottom = current_top + height
        layer_coords.append((top, bottom))
        current_top = bottom

    if visual_width == -1:
        visual_width = total_film_thickness + 2 * inf_display_height
    x_min = -visual_width / 2
    x_max = visual_width / 2
    current_ray_x = 0
    legend_handles = {}

    for i, (layer, theta) in enumerate(zip(layers, angles)):
        top_y, bottom_y = layer_coords[i]
        mat_name = layer.name

        rect_height = abs(top_y - bottom_y)
        rect_y_start = min(top_y, bottom_y)

        rect = patches.Rectangle(
            (x_min, rect_y_start),
            visual_width,
            rect_height,
            linewidth=0,
            facecolor=color_map.get(mat_name, "#CCCCCC"),
            alpha=0.85,
        )
        ax.add_patch(rect)
        if mat_name not in legend_handles:
            legend_handles[mat_name] = rect

        ax.axhline(bottom_y, color="white", lw=0.5, alpha=0.2)

        d_text = "Infinity" if layer.d == float("inf") else f"{layer.d} um"
        label_y = (top_y + bottom_y) / 2
        ax.text(
            x_max * 0.95,
            label_y,
            f"{mat_name}\n{d_text}",
            va="center",
            ha="right",
            fontsize=8,
            color="black",
            weight="bold",
        )

        theta_real = np.real(theta)
        angle_text = f"{np.degrees(theta_real):.1f}°"

        if i == 0:
            arrow_len = min(x_max * 0.3, inf_display_height * 0.8)
            start_x = -arrow_len * np.sin(theta_real)
            start_y = bottom_y - arrow_len * np.cos(theta_real)

            ax.annotate(
                "",
                xy=(0, bottom_y),
                xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle="->", color="black", lw=2),
            )
            ax.text(
                start_x,
                start_y,
                f"Incidence\n{angle_text}",
                color="black",
                fontsize=9,
                va="bottom",
                ha="center",
            )
            current_ray_x = 0
        else:
            layer_h = bottom_y - top_y
            seg_start_x = current_ray_x
            seg_start_y = top_y
            dist_y_remaining = layer_h
            segments = 0
            while dist_y_remaining > 1e-9 and segments < 20:
                segments += 1
                theoretical_end_x = seg_start_x + dist_y_remaining * np.tan(
                    theta_real
                )
                hit_boundary = False
                boundary_x = 0
                if theoretical_end_x > x_max:
                    hit_boundary = True
                    boundary_x = x_max
                elif theoretical_end_x < x_min:
                    hit_boundary = True
                    boundary_x = x_min

                if not hit_boundary:
                    ax.plot(
                        [seg_start_x, theoretical_end_x],
                        [seg_start_y, bottom_y],
                        color="black",
                        lw=1.5,
                    )
                    if segments == 1:
                        mid_x = (seg_start_x + theoretical_end_x) / 2
                        mid_y = (seg_start_y + bottom_y) / 2
                        ax.text(
                            mid_x,
                            mid_y,
                            angle_text,
                            color="black",
                            fontsize=8,
                            ha="left",
                            va="bottom",
                        )
                    current_ray_x = theoretical_end_x
                    dist_y_remaining = 0
                else:
                    dx_to_wall = boundary_x - seg_start_x
                    dy_moved = (
                        dx_to_wall / np.tan(theta_real)
                        if np.tan(theta_real) != 0
                        else 0
                    )
                    hit_y = seg_start_y + dy_moved
                    ax.plot(
                        [seg_start_x, boundary_x],
                        [seg_start_y, hit_y],
                        color="black",
                        lw=1.5,
                    )
                    if segments == 1:
                        ax.text(
                            (seg_start_x + boundary_x) / 2,
                            (seg_start_y + hit_y) / 2,
                            angle_text,
                            color="black",
                            fontsize=8,
                            ha="left",
                        )
                    if boundary_x == x_max:
                        seg_start_x = x_min
                    else:
                        seg_start_x = x_max
                    seg_start_y = hit_y
                    dist_y_remaining -= dy_moved
                    current_ray_x = seg_start_x

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(layer_coords[0][0], layer_coords[-1][1])
    ax.set_xlabel("Lateral Position [Periodic Boundary]")
    ax.set_ylabel("")
    if angle_deg != -1:
        ax.set_title(f"{title} (Incidence: {angle_deg}°)")
    else:
        ax.set_title(title)

    handles, labels = plt.gca().get_legend_handles_labels()
    for name, rect in legend_handles.items():
        handles.append(rect)
        labels.append(name)
    ax.legend(handles, labels, loc="lower left", framealpha=0.9)
    plt.tight_layout()
    return fig


def build_layers_for_visualization(
    material_names: List[str],
    nk_list: List[complex],
    thickness_list: List[float],
) -> List[MockFilm]:
    """
    从材料名、折射率、厚度列表构建 MockFilm 列表（首尾为半无限层）。
    """
    layers = []
    for i, name in enumerate(material_names):
        mat = MockMaterial(name, nk_list[i])
        d = (
            float("inf")
            if i == 0 or i == len(material_names) - 1
            else thickness_list[i]
        )
        layers.append(MockFilm(mat, d))
    return layers


def plot_tmm_filmstack(
    material_film_list: List[Any],
    material_film_name_list: List[str],
    angle_deg: float = 45,
    visual_width: float = -1,
    inf_display_height: float = 100,
) -> plt.Figure:
    """
    从 TMM 层列表与名称列表绘制膜系图，返回 Figure。无 st 依赖。
    """
    degree = np.pi / 180
    th_0 = angle_deg * degree

    def make_mock_material(i: int) -> MockMaterial:
        return MockMaterial(
            material_film_name_list[i], material_film_list[i].nk
        )

    top = make_mock_material(0)
    substrate = make_mock_material(-1)
    layers = [MockFilm(top, float("inf"))]
    for i in range(1, len(material_film_list) - 1):
        layers.append(
            MockFilm(
                make_mock_material(i), material_film_list[i].depth
            )
        )
    layers.append(MockFilm(substrate, float("inf")))

    dir_list = calculate_angles(layers, th_0)
    color_map: Dict[str, Tuple[float, float, float]] = {}
    nmin = min(np.real(l.nk) for l in layers)
    nmax = max(np.real(l.nk) for l in layers)
    k_max = max(np.imag(l.nk) for l in layers)
    for layer in layers:
        if layer.name in color_map:
            continue
        color_map[layer.name] = nk_to_color(
            np.real(layer.nk), np.imag(layer.nk), nmin, nmax, k_max
        )
    return plot_periodic_structure(
        layers,
        dir_list,
        color_map=color_map,
        angle_deg=angle_deg,
        title="TMM filmstack",
        visual_width=visual_width,
        inf_display_height=inf_display_height,
    )
