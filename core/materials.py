"""
材料 n/k 计算逻辑。materials_db 的 value 为 database_material（或 Vacuum 的 material_s）。
"""

from typing import Dict, Any


def get_nk_at_wavelength(
    materials_db: Dict[str, Any],
    name: str,
    wl_um: float,
) -> complex:
    """根据材料库与波长返回复折射率 n + 1j*k。"""
    if name == "Vacuum":
        return 1.0 + 0.0j
    mat = materials_db.get(name)
    if mat is None:
        return 1.0 + 0.0j
    nk = mat.nk_at_wavelength_um(float(wl_um))
    if callable(getattr(nk, "real", None)):
        return complex(nk.real(), nk.imag())
    return complex(nk.real, nk.imag)
