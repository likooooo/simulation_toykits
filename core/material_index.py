"""
材料索引 CSV 的解析与名称清洗。纯函数，不依赖全局变量。
支持缓存：解析结果写入 CSV 同级目录的 .cache 文件，下次直接加载。
"""

import os
import re
import pandas as pd
from typing import Optional


_RE_SUB = re.compile(r"<sub>(\d+)</sub>")
_RE_SUP = re.compile(r"<sup>.*?</sup>")

_CACHE_SUFFIX = ".cache.pkl"


def clean_name(name: Optional[str]) -> Optional[str]:
    """去除 HTML 下标/上标标签，用于显示名。"""
    if not isinstance(name, str):
        return name
    name = _RE_SUB.sub(lambda m: m.group(1), name)
    name = _RE_SUP.sub("", name)
    return name


def load_material_index(csv_path: str) -> pd.DataFrame:
    """
    加载材料索引 CSV 并规范化 book_name、page_name 列。

    :param csv_path: materials_index.csv 的路径
    :return: 含 shelf_name, shelf_id, book_name, book_id, page_name, page_id 等列的 DataFrame
    """
    df = pd.read_csv(csv_path)

    def optimize_column(series):
        unique_names = {name: clean_name(name) for name in series.unique()}
        return series.map(unique_names)

    df["book_name"] = optimize_column(df["book_name"])
    df["page_name"] = optimize_column(df["page_name"])
    return df


def _cache_path_for_csv(csv_path: str) -> str:
    """CSV 同级目录下的缓存文件路径。"""
    base = os.path.splitext(csv_path)[0]
    return base + _CACHE_SUFFIX


def load_material_index_cached(csv_path: str) -> pd.DataFrame:
    """
    加载材料索引：若存在同级目录下的 .cache.pkl 且比 CSV 新，则从缓存加载；
    否则从 CSV 解析并写入缓存。

    :param csv_path: materials_index.csv 的路径
    :return: 含 shelf_name, shelf_id, book_name, book_id, page_name, page_id 等列的 DataFrame
    """
    cache_path = _cache_path_for_csv(csv_path)
    csv_mtime = os.path.getmtime(csv_path) if os.path.isfile(csv_path) else 0
    if os.path.isfile(cache_path) and os.path.getmtime(cache_path) >= csv_mtime:
        return pd.read_pickle(cache_path)
    df = load_material_index(csv_path)
    try:
        df.to_pickle(cache_path)
    except OSError:
        pass
    return df
