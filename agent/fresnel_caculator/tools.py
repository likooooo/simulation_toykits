"""Fresnel 专家智能体工具层：封装 filmstack_simulation 与 simulation_database。"""

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

import simulation  # noqa: F401


def _filmstack_plugin():
    import filmstack_visualizer

    return filmstack_visualizer


def _coerce_path_keys(path_keys: Union[List[str], str, None]) -> List[str]:
    if path_keys is None:
        return []
    if isinstance(path_keys, list):
        return [str(p).strip() for p in path_keys if str(p).strip()]
    text = str(path_keys).strip()
    if not text:
        return []
    if " > " in text:
        return [p.strip() for p in text.split(" > ") if p.strip()]
    return [text]


def _is_air_path(path_keys: List[str]) -> bool:
    return len(path_keys) == 1 and path_keys[0].lower() == "air"


def _air_nk_payload() -> Dict[str, Any]:
    return {
        "material": "air",
        "wavelength_um": [0.0, 1.0],
        "n": [1.0, 1.0],
        "k": [0.0, 0.0],
    }


def list_material_index(
    material_name: str,
) -> Dict[str, Any]:
    """在 simulation_database materials 树中搜索材料名，返回匹配 query path。"""
    key = (material_name or "").strip()
    if not key:
        return {"error": "material_name 不能为空"}

    import simulation_database_parser as sdp

    from simulation_database.database_ui import breadcrumb_for, search_material_paths

    sim_db = sdp.get_simulation_database(init=True)
    paths = search_material_paths(sim_db, key)
    if not paths:
        return {"matches": [], "message": f"未找到材料: {material_name}"}
    return {
        "matches": [
            {"path_keys": p, "label": breadcrumb_for(p)}
            for p in paths[:20]
        ]
    }


def get_material_nk(
    path_keys: Union[List[str], str],
    ratio: float = 1,
    materials_db: Optional[Dict[str, Any]] = None,
    material_token: Optional[str] = None,
) -> Dict[str, Any]:
    """获取指定材料的 n/k 数据。path_keys 为 release 树完整路径；air 可用 [\"air\"]。"""
    keys = _coerce_path_keys(path_keys)
    if not keys:
        return {"error": "path_keys 不能为空"}

    try:
        ratio = float(ratio) if ratio is not None else 1.0
    except (TypeError, ValueError):
        ratio = 1.0
    if not (0 < ratio <= 1):
        return {"error": "ratio 取值范围为 0 < ratio <= 1"}

    if _is_air_path(keys):
        return _air_nk_payload()

    from simulation_database.database_ui import material_nk_arrays

    mat = None
    token = material_token or keys[-1].removesuffix(".yml").removesuffix(".yaml")
    if materials_db and token in materials_db:
        mat = materials_db[token]
    else:
        import simulation_database_parser as sdp

        try:
            mat = sdp.read_at_query_path(sdp.get_simulation_database(init=True), keys)
        except Exception:
            mat = None

    if mat is None:
        return {"error": f"未找到材料: {' > '.join(keys)}", "path_keys": keys}

    try:
        wl, n, k = material_nk_arrays(mat)
        wl = np.asarray(wl)
        n = np.asarray(n)
        k = np.asarray(k)
        if len(wl) == 0:
            return {"material": token, "path_keys": keys, "wavelength_um": [], "n": [], "k": []}
        if ratio < 1.0:
            n_keep = max(1, int(round(len(wl) * ratio)))
            indices = np.linspace(0, len(wl) - 1, n_keep, dtype=int)
            wl, n, k = wl[indices], n[indices], k[indices]
        return {
            "material": getattr(mat, "name", token),
            "path_keys": keys,
            "wavelength_um": wl.tolist(),
            "n": n.tolist(),
            "k": k.tolist(),
        }
    except Exception as e:
        return {"error": str(e), "path_keys": keys}


def export_nk_to_csv(path_keys: Union[List[str], str], out_path: str) -> Dict[str, Any]:
    """将材料 nk 导出为 CSV 文件。"""
    data = get_material_nk(path_keys)
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


def parse_film_formula(
    formula: str,
    materials_db: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Parse formula with bookend expansion (same chain as compute_filmstack)."""
    try:
        import simulation

        fv = _filmstack_plugin()
        materials, thicknesses_um, _ = fv.resolve_formula_stack(
            formula, materials_db or {}, simulation_module=simulation
        )
        layers = [
            {
                "Material": fv.material_unique_name(mat),
                "Thickness_um": float(th),
            }
            for mat, th in zip(materials, thicknesses_um)
        ]
        return {"layers": layers, "formula": formula}
    except Exception as e:
        return {"error": str(e), "formula": formula}


def _resolve_stack(formula: str, materials_db: Optional[Dict[str, Any]]):
    from filmstack_simulation.simulation import resolve_stack_with_layers

    return resolve_stack_with_layers(formula, materials_db)


def compute_filmstack(
    formula: str,
    angle_deg: float,
    wl_um: float,
    materials_db: Optional[Dict[str, Any]] = None,
    out_figure_path: Optional[str] = None,
) -> Dict[str, Any]:
    from filmstack_simulation.simulation import compute_rt_and_coefficients

    try:
        materials, thicknesses, layers = _resolve_stack(formula, materials_db)
    except Exception as e:
        return {"error": str(e), "formula": formula}
    if len(materials) < 2:
        return {"error": "至少需要两层（入射介质与基底）", "formula": formula}

    th_rad = np.deg2rad(angle_deg)
    try:
        (R_s, T_s, R_p, T_p), (r_s, t_s, r_p, t_p) = compute_rt_and_coefficients(
            layers, th_rad, wl_um
        )
    except Exception as e:
        return {"error": str(e), "formula": formula}

    fv = _filmstack_plugin()
    filmstack_fig = fv.plot_filmstack(layers, show=False)

    out = {
        "formula": formula,
        "angle_deg": angle_deg,
        "wl_um": wl_um,
        "R_s": R_s,
        "T_s": T_s,
        "R_p": R_p,
        "T_p": T_p,
        "r_s": str(r_s),
        "t_s": str(t_s),
        "r_p": str(r_p),
        "t_p": str(t_p),
    }
    if out_figure_path:
        try:
            filmstack_fig.savefig(out_figure_path, dpi=150, bbox_inches="tight")
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
) -> Dict[str, Any]:
    from filmstack_simulation.sweep import compute_angle_vs_RT_data

    out_figure_path = out_figure_path or out_figure_rt_path
    if angles_deg is None:
        angles_deg = np.linspace(0, 89, 90).tolist()
    angles_arr = np.asarray(angles_deg, dtype=float)

    try:
        materials, thicknesses, layers = _resolve_stack(formula, materials_db)
    except Exception as e:
        return {"error": str(e), "formula": formula}
    if len(materials) < 2:
        return {"error": "至少需要两层", "formula": formula}

    data = compute_angle_vs_RT_data(layers, wl_um, angles_arr)
    rs, rp, ts, tp = data["R_s"], data["R_p"], data["T_s"], data["T_p"]

    fig = _filmstack_plugin().plot_RT_vs_angle(angles_arr, rs, rp, ts, tp, wl_um)
    figure_saved = None
    if out_figure_path:
        try:
            fig.savefig(out_figure_path, dpi=150, bbox_inches="tight")
            figure_saved = os.path.abspath(out_figure_path)
        except Exception:
            pass
    return {
        "formula": formula,
        "wl_um": wl_um,
        "angles_deg": angles_arr.tolist(),
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
    from filmstack_simulation.sweep import compute_wavelength_vs_RT_data

    wls = np.linspace(wl_min_um, wl_max_um, num_points)
    try:
        materials, thicknesses, layers = _resolve_stack(formula, materials_db)
    except Exception as e:
        return {"error": str(e), "formula": formula}
    if len(materials) < 2:
        return {"error": "至少需要两层", "formula": formula}

    fv = _filmstack_plugin()
    data = compute_wavelength_vs_RT_data(layers, wls, angle_deg)
    rs, rp, ts, tp = data["R_s"], data["R_p"], data["T_s"], data["T_p"]

    fig_rt = fv.plot_RT_vs_wavelength(wls, rs, rp, ts, tp, angle_deg)
    fig_nk = fv.plot_filmstack_material_nk_1x2(fv.unique_materials_in_stack(materials), wls)
    out = {
        "formula": formula,
        "angle_deg": angle_deg,
        "wl_range": [wl_min_um, wl_max_um],
        "num_points": num_points,
    }
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
