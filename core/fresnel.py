"""Fresnel / TMM：传输矩阵法反射透射率，实现来自 simulation.so。"""

import numpy as np
from typing import List, Tuple, Any, Callable

import simulation_loader
simulation_loader.get_simulation_module()

from assets.simulation import (
    meterial_s,
    TMM_propagate_direction,
    TMM_interface_transfer_matrix_with_thickness_s,
    TMM_interface_transfer_matrix_with_thickness_p,
    TMM_get_r_t_from_tmm,
    TMM_get_r_t_power_from_tmm_s,
    TMM_get_r_t_power_from_tmm_p,
)


def build_tmm_layers(
    material_factory: Callable[[], Any],
    nk_list: List[complex],
    thickness_list: List[float],
) -> List[Any]:
    """
    根据折射率列表和厚度列表构建 TMM 所需的层列表（首尾为入射与基底，中间为薄膜）。

    :param material_factory: 无参调用返回具有 .nk、.depth 属性的对象（如 meterial_s）
    :param nk_list: 每层的复折射率
    :param thickness_list: 每层厚度 (μm)，首尾通常为 0（半无限）
    :return: 供 TMM_* 使用的层列表
    """
    upper = material_factory()
    upper.nk = nk_list[0]
    substrate = material_factory()
    substrate.nk = nk_list[-1]

    def make_film(nk: complex, depth: float) -> Any:
        m = material_factory()
        m.nk = nk
        m.depth = depth
        return m

    layers = [upper]
    for i in range(1, len(thickness_list) - 1):
        layers.append(make_film(nk_list[i], thickness_list[i]))
    layers.append(substrate)
    return layers


def compute_RT(
    layers: List[Any],
    th0_rad: float,
    wl_um: float,
) -> Tuple[float, float, float, float]:
    """
    计算 TE/TM 的反射率与透射率 R_s, T_s, R_p, T_p。

    :param layers: build_tmm_layers 返回的层列表
    :param th0_rad: 入射角 (弧度)
    :param wl_um: 波长 (μm)
    :return: (R_s, T_s, R_p, T_p)
    """
    dir_list = TMM_propagate_direction(layers, th0_rad)
    tmm_s = TMM_interface_transfer_matrix_with_thickness_s(layers, dir_list, wl_um)
    tmm_p = TMM_interface_transfer_matrix_with_thickness_p(layers, dir_list, wl_um)
    R_s, T_s = TMM_get_r_t_power_from_tmm_s(
        tmm_s[-1], layers[0].nk, dir_list[0], layers[-1].nk, dir_list[-1]
    )
    R_p, T_p = TMM_get_r_t_power_from_tmm_p(
        tmm_p[-1], layers[0].nk, dir_list[0], layers[-1].nk, dir_list[-1]
    )
    return R_s, T_s, R_p, T_p


def get_r_t(
    layers: List[Any],
    th0_rad: float,
    wl_um: float,
) -> Tuple[complex, complex, complex, complex]:
    """
    计算 Fresnel 系数 r_s, t_s, r_p, t_p。
    """
    dir_list = TMM_propagate_direction(layers, th0_rad)
    tmm_s = TMM_interface_transfer_matrix_with_thickness_s(layers, dir_list, wl_um)
    tmm_p = TMM_interface_transfer_matrix_with_thickness_p(layers, dir_list, wl_um)
    r_s, t_s = TMM_get_r_t_from_tmm(tmm_s[-1])
    r_p, t_p = TMM_get_r_t_from_tmm(tmm_p[-1])
    return r_s, t_s, r_p, t_p
