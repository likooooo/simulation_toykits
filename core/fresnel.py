"""Fresnel / TMM：传输矩阵法反射透射率，实现来自 simulation.so。"""

from typing import Any, List, Tuple

from core import simulation_loader

SUBSTRATE_DEPTH_UM = 1e9


def _sim():
    return simulation_loader.get_simulation_module()


def _cplx(z) -> complex:
    if callable(getattr(z, "real", None)):
        return complex(z.real(), z.imag())
    return complex(z.real, z.imag)


def layer_nk_at(layer, wl_um: float) -> complex:
    return _cplx(layer.background_material.nk_at_wavelength_um(float(wl_um)))


def set_layer_nk(layer, nk, name: str = "") -> None:
    sim = _sim()
    layer.background_material = sim.material_s.from_nk(complex(nk), name)


def build_tmm_layers(
    nk_list: List[complex],
    thickness_list: List[float],
) -> List[Any]:
    """根据折射率与厚度构建 TMM 层列表（首尾为入射介质与基底）。"""
    sim = _sim()
    upper = sim.make_layer_from_nk_s(complex(nk_list[0]), float(thickness_list[0]), "upper")
    substrate = sim.make_layer_from_nk_s(
        complex(nk_list[-1]), float(SUBSTRATE_DEPTH_UM), "substrate"
    )
    layers = [upper]
    for i in range(1, len(thickness_list) - 1):
        layers.append(
            sim.make_layer_from_nk_s(
                complex(nk_list[i]), float(thickness_list[i]), f"layer_{i}"
            )
        )
    layers.append(substrate)
    return layers


def compute_RT(
    layers: List[Any],
    th0_rad: float,
    wl_um: float,
) -> Tuple[float, float, float, float]:
    """计算 TE/TM 反射率与透射率 R_s, T_s, R_p, T_p。"""
    sim = _sim()
    wl_um = float(wl_um)
    dir_list = sim.TMM_propagate_direction_s(layers, th0_rad, wl_um)
    tmm_s = sim.TMM_interface_transfer_matrix_with_thickness_s(layers, dir_list, wl_um)
    tmm_p = sim.TMM_interface_transfer_matrix_with_thickness_p(layers, dir_list, wl_um)
    R_s, T_s = sim.TMM_get_r_t_power_from_tmm_s(
        tmm_s[-1],
        layer_nk_at(layers[0], wl_um),
        dir_list[0],
        layer_nk_at(layers[-1], wl_um),
        dir_list[-1],
    )
    R_p, T_p = sim.TMM_get_r_t_power_from_tmm_p(
        tmm_p[-1],
        layer_nk_at(layers[0], wl_um),
        dir_list[0],
        layer_nk_at(layers[-1], wl_um),
        dir_list[-1],
    )
    return float(R_s), float(T_s), float(R_p), float(T_p)


def get_r_t(
    layers: List[Any],
    th0_rad: float,
    wl_um: float,
) -> Tuple[complex, complex, complex, complex]:
    """计算 Fresnel 系数 r_s, t_s, r_p, t_p。"""
    sim = _sim()
    wl_um = float(wl_um)
    dir_list = sim.TMM_propagate_direction_s(layers, th0_rad, wl_um)
    tmm_s = sim.TMM_interface_transfer_matrix_with_thickness_s(layers, dir_list, wl_um)
    tmm_p = sim.TMM_interface_transfer_matrix_with_thickness_p(layers, dir_list, wl_um)
    r_s, t_s = sim.TMM_get_r_t_from_tmm(tmm_s[-1])
    r_p, t_p = sim.TMM_get_r_t_from_tmm(tmm_p[-1])
    return _cplx(r_s), _cplx(t_s), _cplx(r_p), _cplx(t_p)
