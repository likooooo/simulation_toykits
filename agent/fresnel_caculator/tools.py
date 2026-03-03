"""Fresnel 专家智能体工具层：封装 core 与 refractiveindex。需从仓库根运行以加载 simulation.so。"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _db_path():
    return os.path.join(_REPO_ROOT, "assets", "refractiveindex.info-database")


def _load_nk_standalone(shelf_id: str, book_id: str, page_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 refractiveindex 数据库加载材料 nk，返回 (波长 um, n, k)。不依赖 Streamlit。"""
    from core.refractiveindex import RefractiveIndex
    db_path = _db_path()
    if not os.path.isdir(db_path):
        raise FileNotFoundError(f"材料数据库不存在: {db_path}")
    ri = RefractiveIndex(databasePath=db_path, auto_download=False)
    material = ri.getMaterial(shelf=shelf_id, book=book_id, page=page_id)
    wl_um = material.originalData["wavelength (um)"]
    wl_nm = wl_um * 1e3
    try:
        n = material.getRefractiveIndex(wl_nm)
    except Exception:
        n = np.ones_like(wl_um)
    try:
        k = material.getExtinctionCoefficient(wl_nm)
    except Exception:
        k = np.zeros_like(wl_um)
    return np.asarray(wl_um), np.asarray(n), np.asarray(k)


def list_material_index(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """
    列出材料索引（shelf / book / page）。返回可被 JSON 序列化的结构。
    :param csv_path: 可选，默认使用 assets/.../materials_index.csv
    """
    if csv_path is None:
        csv_path = os.path.join(_db_path(), "materials_index.csv")
    if not os.path.isfile(csv_path):
        return {"error": f"索引文件不存在: {csv_path}", "shelves": []}
    from core.material_index import load_material_index_cached
    df = load_material_index_cached(csv_path)
    if df.empty:
        return {"shelves": [], "message": "索引为空"}
    shelves = []
    for shelf_name in df["shelf_name"].drop_duplicates().head(20):
        sub = df[df["shelf_name"] == shelf_name]
        books = []
        for _, row in sub.drop_duplicates("book_id").head(30).iterrows():
            books.append({
                "book_name": row["book_name"],
                "book_id": row["book_id"],
                "page_id": row["page_id"],
                "page_name": row["page_name"],
            })
        shelves.append({"shelf_name": shelf_name, "shelf_id": sub["shelf_id"].iloc[0], "books": books})
    return {"shelves": shelves, "csv_path": csv_path}


def get_material_nk(shelf_id: str, book_id: str, page_id: str) -> Dict[str, Any]:
    """
    获取指定材料的 n/k 数据（波长 um, n, k 数组）。Vacuum 返回 1+0j。
    """
    if book_id == "Vacuum" or book_id.strip() == "Vacuum":
        return {"wavelength_um": [0.0, 1.0], "n": [1.0, 1.0], "k": [0.0, 0.0], "material": "Vacuum"}
    try:
        wl, n, k = _load_nk_standalone(shelf_id, book_id, page_id)
        return {
            "material": book_id,
            "wavelength_um": wl.tolist(),
            "n": np.asarray(n).tolist(),
            "k": np.asarray(k).tolist(),
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
    """
    解析多层膜公式，得到层列表。支持 (Material thickness)^N 与 Material thickness [n k]。
    返回每层的 Material, Thickness (um), 以及可选的 n, k。
    """
    from core.formula import parse_formula_v1
    try:
        layers = parse_formula_v1(formula)
        return {"layers": layers, "formula": formula}
    except Exception as e:
        return {"error": str(e), "formula": formula}


def _materials_db_from_index_row(shelf_id: str, book_id: str, page_id: str, material_name: str) -> Dict[str, Dict]:
    """单条材料构成 materials_db 的一项（key 为 material_name）。"""
    return {
        material_name: {
            "Shelf ID": shelf_id,
            "Book ID": book_id,
            "Page ID": page_id,
            "Material Name": material_name,
        }
    }


def layers_to_nk_list(
    layers: List[Dict],
    wl_um: float,
    materials_db: Optional[Dict[str, Dict]] = None,
) -> Tuple[List[complex], Optional[str]]:
    """
    根据层配置与波长计算每层 nk。materials_db 的 key 为材料名，value 含 Shelf ID, Book ID, Page ID。
    返回 (nk_list, error_message)。若某材料不在库中且无 n,k 则用 1+0j。
    """
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
        if name == "Vacuum":
            nk_list.append(1.0 + 0.0j)
            continue
        try:
            nk = get_nk_at_wavelength(
                materials_db, name, wl_um,
                lambda s, b, p: _load_nk_standalone(s, b, p),
            )
            nk_list.append(nk)
        except Exception:
            nk_list.append(1.0 + 0.0j)
    return nk_list, None


def compute_filmstack(
    formula: str,
    angle_deg: float,
    wl_um: float,
    materials_db: Optional[Dict[str, Dict]] = None,
    out_figure_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    解析公式、计算单波长单角度下的 R/T 与膜系图。
    materials_db: 材料名 -> {Shelf ID, Book ID, Page ID}，用于从数据库取 nk。
    若某层材料名不在 materials_db 且公式中无 n k，则用 1+0j。
    """
    from core import simulation_loader
    simulation_loader.get_simulation_module()
    from simulation import meterial_s
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

    def load_nk(shelf_id: str, book_id: str, page_id: str):
        if book_id == "Vacuum":
            return np.array([0, 100]), np.array([1, 1]), np.array([0, 0])
        return _load_nk_standalone(shelf_id, book_id, page_id)

    df = pd.DataFrame(layers)
    df = with_nk_columns(df, wl_um, lambda name: get_nk_at_wavelength(materials_db, name, wl_um, load_nk))
    names = df["Material"].tolist()
    nk_list = [n + 1j * k for n, k in zip(df["n"].tolist(), df["k"].tolist())]
    thickness_list = df["Thickness (um)"].tolist()

    try:
        result = compute_fresnel_and_filmstack(
            material_factory=lambda: meterial_s(),
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
    materials_db: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Any]:
    """
    批量计算多组膜系配置的 R/T，用于设计时筛选。返回每条公式的 R_s, T_s, R_p, T_p 及 error（若有）。
    """
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
    materials_db: Optional[Dict[str, Dict]] = None,
    angles_deg: Optional[List[float]] = None,
    out_figure_path: Optional[str] = None,
) -> Dict[str, Any]:
    """固定波长，计算 R/T 随角度的变化，并可选保存曲线图。"""
    from core import simulation_loader
    simulation_loader.get_simulation_module()
    from core.formula import parse_formula_v1
    from core.materials import with_nk_columns, get_nk_at_wavelength
    from core.fresnel import build_tmm_layers, compute_RT
    from simulation import meterial_s
    from core.spectral import compute_angle_vs_RT_figures

    if materials_db is None:
        materials_db = {}
    if angles_deg is None:
        angles_deg = np.linspace(0, 89, 90).tolist()
    angles_deg = np.asarray(angles_deg)

    parsed = parse_film_formula(formula)
    if "error" in parsed:
        return parsed
    layers_data = parsed["layers"]
    if len(layers_data) < 2:
        return {"error": "至少需要两层", "formula": formula}

    def load_nk(s, b, p):
        if b == "Vacuum":
            return np.array([0, 100]), np.array([1, 1]), np.array([0, 0])
        return _load_nk_standalone(s, b, p)

    df = with_nk_columns(pd.DataFrame(layers_data), wl_um, lambda n: get_nk_at_wavelength(materials_db, n, wl_um, load_nk))
    nk_list = [n + 1j * k for n, k in zip(df["n"].tolist(), df["k"].tolist())]
    thickness_list = df["Thickness (um)"].tolist()
    tmm_layers = build_tmm_layers(lambda: meterial_s(), nk_list, thickness_list)

    try:
        figs = compute_angle_vs_RT_figures(tmm_layers, wl_um, angles_deg)
    except Exception as e:
        return {"error": str(e), "formula": formula}
    if out_figure_path and figs:
        figs[0].savefig(out_figure_path, dpi=150, bbox_inches="tight")
    return {
        "formula": formula,
        "wl_um": wl_um,
        "angles_deg": angles_deg.tolist(),
        "figure_saved": out_figure_path if (out_figure_path and figs) else None,
    }


def compute_wavelength_vs_rt(
    formula: str,
    angle_deg: float,
    wl_min_um: float,
    wl_max_um: float,
    num_points: int = 100,
    materials_db: Optional[Dict[str, Dict]] = None,
    out_figure_rt_path: Optional[str] = None,
    out_figure_nk_path: Optional[str] = None,
) -> Dict[str, Any]:
    """固定角度，计算 R/T 随波长的变化，可选保存 R/T 曲线图与 n-k 曲线图。"""
    from core import simulation_loader
    simulation_loader.get_simulation_module()
    from core.formula import parse_formula_v1
    from core.materials import with_nk_columns, get_nk_at_wavelength
    from core.fresnel import build_tmm_layers
    from simulation import meterial_s
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
    def load_nk(s, b, p):
        if b == "Vacuum":
            return np.array([0, 100]), np.array([1, 1]), np.array([0, 0])
        return _load_nk_standalone(s, b, p)
    df = with_nk_columns(pd.DataFrame(layers_data), wl_center, lambda n: get_nk_at_wavelength(materials_db, n, wl_center, load_nk))
    names = df["Material"].tolist()
    n_col = df["n"].tolist()
    k_col = df["k"].tolist()
    nk_list_0 = [n + 1j * k for n, k in zip(n_col, k_col)]
    thickness_list = df["Thickness (um)"].tolist()
    tmm_layers = build_tmm_layers(lambda: meterial_s(), nk_list_0, thickness_list)
    nk_map, _ = build_nk_map_for_wavelengths(names, n_col, k_col, wls, materials_db, lambda name, w: get_nk_at_wavelength(materials_db, name, w, load_nk))

    try:
        fig_rt, fig_nk = compute_wavelength_vs_RT_figures(tmm_layers, names, nk_map, wls, angle_deg)
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
    """将结果列表（如多组膜系及其 R/T）保存为 CSV。"""
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
    """将 numpy 等类型转为 JSON 可序列化。"""
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
    """执行指定工具并返回可 JSON 序列化的结果。"""
    if name not in TOOLS:
        return {"error": f"未知工具: {name}", "available": list(TOOLS.keys())}
    fn = TOOLS[name]
    try:
        result = fn(**arguments)
        return _to_json_safe(result) if isinstance(result, (dict, list)) else result
    except Exception as e:
        return {"error": str(e), "tool": name}
