"""
材料 n/k 计算逻辑。所有参数通过函数传入，不依赖全局变量。
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, Any


def get_nk_at_wavelength(
    materials_db: Dict[str, Any],
    name: str,
    wl_um: float,
    load_nk_func: Callable[[str, str, str], tuple],
) -> complex:
    """
    根据材料数据库和波长返回复折射率 n + 1j*k。

    :param materials_db: 材料数据库字典，key 为材料名，value 含 "Shelf ID", "Book ID", "Page ID"
    :param name: 材料名称
    :param wl_um: 波长 (μm)
    :param load_nk_func: 函数 (shelf_id, book_id, page_id) -> (wls, ns, ks) 用于加载 nk 数据
    :return: 复折射率 n + 1j*k，Vacuum 或异常时返回 1+0j
    """
    if name == "Vacuum":
        return 1.0 + 0.0j
    if name not in materials_db:
        return 1.0 + 0.0j
    row = materials_db[name]
    wls, ns, ks = load_nk_func(
        row["Shelf ID"], row["Book ID"], row["Page ID"]
    )
    n_val = np.interp(wl_um, wls, ns)
    k_val = np.interp(wl_um, wls, ks)
    return n_val + 1j * k_val


def with_nk_columns(
    df: pd.DataFrame,
    wl_um: float,
    get_nk_func: Callable[[str], complex],
) -> pd.DataFrame:
    """
    为 DataFrame 补全 n、k 列。若某行已有 n、k 则保留，否则用 get_nk_func(材料名) 计算。

    :param df: 含 "Material" 列，可选 "n", "k"
    :param wl_um: 波长 (μm)，仅用于语义，实际取值由 get_nk_func 决定
    :param get_nk_func: 函数 (material_name) -> n+1j*k
    :return: 带 n, k 列的副本
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
        nk = get_nk_func(row["Material"])
        n_list.append(np.real(nk))
        k_list.append(np.imag(nk))
    df_show["n"] = n_list
    df_show["k"] = k_list
    return df_show
