"""
材料 n/k 计算逻辑。materials_db 的 value 为 database_material（或 Vacuum 的 material_s）。
"""

import pandas as pd
from typing import Callable, Dict, Any


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


def with_nk_columns(
    df: pd.DataFrame,
    wl_um: float,
    get_nk_func: Callable[[str], complex] | None = None,
) -> pd.DataFrame:
    """
    为 DataFrame 补全 n、k 列。若某行已有 n、k 则保留，否则用 get_nk_func(材料名) 或 materials_db 计算。
    """
    if df.empty:
        return df
    df_show = df.copy()
    n_list, k_list = [], []
    for _, row in df.iterrows():
        n_val, k_val = row.get("n"), row.get("k")
        if not (pd.isna(n_val) or pd.isna(k_val)):
            n_list.append(float(n_val))
            k_list.append(float(k_val))
            continue
        if get_nk_func is not None:
            nk = get_nk_func(row["Material"])
        else:
            nk = 1.0 + 0.0j
        n_list.append(float(nk.real))
        k_list.append(float(nk.imag))
    df_show["n"] = n_list
    df_show["k"] = k_list
    return df_show
