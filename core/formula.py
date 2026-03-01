"""
膜系公式解析。纯函数，不依赖全局变量。
"""

import re
from typing import List, Dict, Any


def parse_formula(formula: str) -> List[Dict[str, Any]]:
    """
    支持格式: (H L)^10, (H 0.1 L 0.2)^5, SiO2 0.1 TiO2 0.2
    返回 [{"Material": m, "Thickness (um)": t}, ...]
    """
    while "(" in formula:
        match = re.search(r"\(([^()]+)\)\^(\d+)", formula)
        if not match:
            break
        content, times = match.group(1), int(match.group(2))
        formula = formula.replace(match.group(0), (content + " ") * times)

    pairs = re.findall(r"(\S+)\s+([\d.]+)", formula)
    return [{"Material": m, "Thickness (um)": float(t)} for m, t in pairs]


def parse_formula_v1(formula: str) -> List[Dict[str, Any]]:
    """
    支持语法：
    - Material Thickness
    - Material Thickness n k
    - 括号周期：(SiO2 0.1 1.5 0.001 Ta2O5 0.01)^5

    返回 [{"Material": m, "Thickness (um)": t, "n": n_or_None, "k": k_or_None}, ...]
    """
    while "(" in formula:
        match = re.search(r"\(([^()]+)\)\^(\d+)", formula)
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

        layers.append({
            "Material": material,
            "Thickness (um)": thickness,
            "n": n_override,
            "k": k_override,
        })
    return layers
