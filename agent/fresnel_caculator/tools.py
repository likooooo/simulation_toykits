"""Fresnel 专家智能体工具层：封装 core 与 simulation_database。需从仓库根运行以加载 simulation.so。"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _material_nk_arrays(mat) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from core.simulation_database_ui import material_nk_arrays
    return material_nk_arrays(mat)


def list_material_index(
    material_name: str,
    csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    """在 simulation_database materials 树中搜索材料名，返回匹配路径。"""
    key = (material_name or "").strip()
    if not key:
        return {"error": "material_name 不能为空"}

    from core.simulation_database_ui import ensure_simulation_database_initialized, search_material_paths

    sim_db = ensure_simulation_database_initialized()
    paths = search_material_paths(sim_db, key, db_name="materials")
    if not paths:
        return {
            "shelf_id": "materials",
            "books": [],
            "message": f"未找到材料: {material_name}",
        }
    books = [{"page_id": " > ".join(p), "path": p} for p in paths[:20]]
    return {"shelf_id": "materials", "books": books}


def get_material_nk(
    shelf_id: str,
    book_id: str,
    page_id: str,
    ratio: float = 1,
    materials_db: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """获取指定材料的 n/k 数据。优先 materials_db[book_id]，否则按 path 从数据库读取。"""
    try:
        ratio = float(ratio) if ratio is not None else 1.0
    except (TypeError, ValueError):
        ratio = 1.0
    if not (0 < ratio <= 1):
        return {"error": "ratio 取值范围为 0 < ratio <= 1"}

    if book_id == "Vacuum" or str(book_id).strip() == "Vacuum":
        return {"wavelength_um": [0.0, 1.0], "n": [1.0, 1.0], "k": [0.0, 0.0], "material": "Vacuum"}

    mat = None
    if materials_db and book_id in materials_db:
        mat = materials_db[book_id]
    else:
        from core.simulation_database_ui import (
            ensure_simulation_database_initialized,
            read_material_by_path,
        )
        sim_db = ensure_simulation_database_initialized()
        path_keys = page_id.split(" > ") if " > " in (page_id or "") else [book_id]
        if page_id and page_id.startswith("[") is False and " > " in page_id:
            path_keys = [p.strip() for p in page_id.split(" > ")]
        elif materials_db is None:
            path_keys = [book_id] if not page_id or page_id == book_id else [book_id, page_id]
        try:
            mat = read_material_by_path(sim_db, shelf_id or "materials", path_keys)
        except Exception:
            mat = None

    if mat is None:
        return {"error": f"未找到材料: {book_id}", "shelf_id": shelf_id, "book_id": book_id, "page_id": page_id}

    try:
        wl, n, k = _material_nk_arrays(mat)
        wl = np.asarray(wl)
        n = np.asarray(n)
        k = np.asarray(k)
        if len(wl) == 0:
            return {"material": book_id, "wavelength_um": [], "n": [], "k": []}
        if ratio < 1.0:
            n_keep = max(1, int(round(len(wl) * ratio)))
            indices = np.linspace(0, len(wl) - 1, n_keep, dtype=int)
            wl, n, k = wl[indices], n[indices], k[indices]
        return {
            "material": getattr(mat, "name", book_id),
            "wavelength_um": wl.tolist(),
            "n": n.tolist(),
            "k": k.tolist(),
        }
    except Exception as e:
        return {"error": str(e), "shelf_id": shelf_id, "book_id": book_id, "page_id": page_id}


def export_nk_to_csv(shelf_id: str, book_id: str, page_id: str, out_path: str) -> Dict[str, Any]:
    """将材料 nk 导出为 CSV 文件。"""
    data = get_material_nk(shelf_id, book_id, page_id)
    if "error" in data:
        return data
    df = pd.DataFrame({
        "Wavelength(um)": data["wavelength_um"],
        "n": data["n"],
        "k": data["k"],
    })
    try:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_csv(out_path, index=False)
        return {"success": True, "path": os.path.abspath(out_path), "rows": len(df)}
    except Exception as e:
        return {"error": str(e), "path": out_path}


def parse_film_formula(formula: str) -> Dict[str, Any]:
    from core.formula import parse_formula_v1
    try:
        layers = parse_formula_v1(formula)
        return {"layers": layers, "formula": formula}
    except Exception as e:
        return {"error": str(e), "formula": formula}


def layers_to_nk_list(
    layers: List[Dict],
    wl_um: float,
    materials_db: Optional[Dict[str, Any]] = None,
) -> Tuple[List[complex], Optional[str]]:
    from core.materials import get_nk_at_wavelength
    if materials_db is None:
        materials_db = {}
    nk_list = []
    for layer in layers:
        name = layer.get("Material", "Vacuum")
        n_val, k_val = layer.get("n"), layer.get("k")
        if n_val is not None and k_val is not None:
            nk_list.append(float(n_val) + 1j * float(k_val))
            continue
        nk_list.append(get_nk_at_wavelength(materials_db, name, wl_um))
    return nk_list, None


def compute_filmstack(
    formula: str,
    angle_deg: float,
    wl_um: float,
    materials_db: Optional[Dict[str, Any]] = None,
    out_figure_path: Optional[str] = None,
) -> Dict[str, Any]:
    from core.formula import parse_formula_v1
    from core.materials import with_nk_columns, get_nk_at_wavelength
    from core.films import compute_fresnel_and_filmstack

    if materials_db is None:
        materials_db = {}

    parsed = parse_film_formula(formula)
    if "error" in parsed:
        return parsed
    layers = parsed["layers"]
    if len(layers) < 2:
        return {"error": "至少需要两层（入射介质与基底）", "formula": formula}

    df = pd.DataFrame(layers)
    df = with_nk_columns(
        df, wl_um,
        lambda name: get_nk_at_wavelength(materials_db, name, wl_um),
    )
    names = df["Material"].tolist()
    nk_list = [n + 1j * k for n, k in zip(df["n"].tolist(), df["k"].tolist())]
    thickness_list = df["Thickness (um)"].tolist()

    try:
        result = compute_fresnel_and_filmstack(
            material_names=names,
            nk_list=nk_list,
            thickness_list=thickness_list,
            angle_deg=angle_deg,
            wl_um=wl_um,
        )
    except Exception as e:
        return {"error": str(e), "formula": formula}

    out = {
        "formula": formula,
        "angle_deg": angle_deg,
        "wl_um": wl_um,
        "R_s": result.R_s,
        "T_s": result.T_s,
        "R_p": result.R_p,
        "T_p": result.T_p,
        "r_s": str(result.r_s),
        "t_s": str(result.t_s),
        "r_p": str(result.r_p),
        "t_p": str(result.t_p),
    }
    if out_figure_path:
        try:
            result.filmstack_fig.savefig(out_figure_path, dpi=150, bbox_inches="tight")
            out["figure_path"] = os.path.abspath(out_figure_path)
        except Exception as e:
            out["figure_save_error"] = str(e)
    return out


def compute_filmstack_batch(
    formulas: List[str],
    angle_deg: float,
    wl_um: float,
    materials_db: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    results = []
    for formula in formulas:
        r = compute_filmstack(formula, angle_deg, wl_um, materials_db, out_figure_path=None)
        if "error" in r:
            results.append({"formula": formula, "error": r["error"]})
        else:
            results.append({
                "formula": formula,
                "R_s": r["R_s"],
                "T_s": r["T_s"],
                "R_p": r["R_p"],
                "T_p": r["T_p"],
            })
    return {"angle_deg": angle_deg, "wl_um": wl_um, "results": results}


def compute_angle_vs_rt(
    formula: str,
    wl_um: float,
    materials_db: Optional[Dict[str, Any]] = None,
    angles_deg: Optional[List[float]] = None,
    out_figure_path: Optional[str] = None,
    out_figure_rt_path: Optional[str] = None,
    out_figure_nk_path: Optional[str] = None,
) -> Dict[str, Any]:
    out_figure_path = out_figure_path or out_figure_rt_path or out_figure_nk_path
    if materials_db is None or not isinstance(materials_db, dict):
        materials_db = {}
    from core.formula import parse_formula_v1
    from core.materials import with_nk_columns, get_nk_at_wavelength
    from core.fresnel import build_tmm_layers
    from core.spectral import compute_angle_vs_RT_figures
    if angles_deg is None:
        angles_deg = np.linspace(0, 89, 90).tolist()
    angles_deg = np.asarray(angles_deg)

    parsed = parse_film_formula(formula)
    if "error" in parsed:
        return parsed
    layers_data = parsed["layers"]
    if len(layers_data) < 2:
        return {"error": "至少需要两层", "formula": formula}

    df = with_nk_columns(
        pd.DataFrame(layers_data), wl_um,
        lambda n: get_nk_at_wavelength(materials_db, n, wl_um),
    )
    nk_list = [n + 1j * k for n, k in zip(df["n"].tolist(), df["k"].tolist())]
    thickness_list = df["Thickness (um)"].tolist()
    tmm_layers = build_tmm_layers(nk_list, thickness_list)

    try:
        figs = compute_angle_vs_RT_figures(tmm_layers, wl_um, angles_deg)
    except Exception as e:
        return {"error": str(e), "formula": formula}
    figure_saved = None
    if figs:
        for path in (out_figure_path, out_figure_rt_path, out_figure_nk_path):
            if path:
                try:
                    figs[0].savefig(path, dpi=150, bbox_inches="tight")
                    figure_saved = figure_saved or os.path.abspath(path)
                except Exception:
                    pass
    return {
        "formula": formula,
        "wl_um": wl_um,
        "angles_deg": angles_deg.tolist(),
        "figure_saved": figure_saved,
    }


def compute_wavelength_vs_rt(
    formula: str,
    angle_deg: float,
    wl_min_um: float,
    wl_max_um: float,
    num_points: int = 100,
    materials_db: Optional[Dict[str, Any]] = None,
    out_figure_rt_path: Optional[str] = None,
    out_figure_nk_path: Optional[str] = None,
) -> Dict[str, Any]:
    from core.formula import parse_formula_v1
    from core.materials import with_nk_columns, get_nk_at_wavelength
    from core.fresnel import build_tmm_layers
    from core.spectral import compute_wavelength_vs_RT_figures, build_nk_map_for_wavelengths

    if materials_db is None:
        materials_db = {}
    wls = np.linspace(wl_min_um, wl_max_um, num_points)
    parsed = parse_film_formula(formula)
    if "error" in parsed:
        return parsed
    layers_data = parsed["layers"]
    if len(layers_data) < 2:
        return {"error": "至少需要两层", "formula": formula}

    wl_center = float(np.mean(wls))
    df = with_nk_columns(
        pd.DataFrame(layers_data), wl_center,
        lambda n: get_nk_at_wavelength(materials_db, n, wl_center),
    )
    names = df["Material"].tolist()
    n_col = df["n"].tolist()
    k_col = df["k"].tolist()
    nk_list_0 = [n + 1j * k for n, k in zip(n_col, k_col)]
    thickness_list = df["Thickness (um)"].tolist()
    tmm_layers = build_tmm_layers(nk_list_0, thickness_list)
    nk_map, _ = build_nk_map_for_wavelengths(
        names, n_col, k_col, wls, materials_db,
        lambda name, w: get_nk_at_wavelength(materials_db, name, w),
    )

    try:
        fig_rt, fig_nk, _ = compute_wavelength_vs_RT_figures(tmm_layers, names, nk_map, wls, angle_deg)
    except Exception as e:
        return {"error": str(e), "formula": formula}
    out = {"formula": formula, "angle_deg": angle_deg, "wl_range": [wl_min_um, wl_max_um], "num_points": num_points}
    if out_figure_rt_path:
        fig_rt.savefig(out_figure_rt_path, dpi=150, bbox_inches="tight")
        out["figure_rt_path"] = os.path.abspath(out_figure_rt_path)
    if out_figure_nk_path:
        fig_nk.savefig(out_figure_nk_path, dpi=150, bbox_inches="tight")
        out["figure_nk_path"] = os.path.abspath(out_figure_nk_path)
    return out


def save_results_csv(rows: List[Dict[str, Any]], out_path: str) -> Dict[str, Any]:
    try:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        return {"success": True, "path": os.path.abspath(out_path), "rows": len(rows)}
    except Exception as e:
        return {"error": str(e), "path": out_path}


TOOLS = {
    "list_material_index": list_material_index,
    "get_material_nk": get_material_nk,
    "export_nk_to_csv": export_nk_to_csv,
    "parse_film_formula": parse_film_formula,
    "compute_filmstack": compute_filmstack,
    "compute_filmstack_batch": compute_filmstack_batch,
    "compute_angle_vs_rt": compute_angle_vs_rt,
    "compute_wavelength_vs_rt": compute_wavelength_vs_rt,
    "save_results_csv": save_results_csv,
}


def _to_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


def run_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if name not in TOOLS:
        return {"error": f"未知工具: {name}", "available": list(TOOLS.keys())}
    fn = TOOLS[name]
    try:
        result = fn(**arguments)
        return _to_json_safe(result) if isinstance(result, (dict, list)) else result
    except Exception as e:
        return {"error": str(e), "tool": name}
