"""Build optimization JSON from Freehand session state."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from filmstack_simulation.filmstack_optimization.local_search.freehand_state import build_freehand_wl_indices
from filmstack_simulation.filmstack_optimization.local_search.opt_config import get_freehand_cost_scope, load_freehand_base_config
import filmstack_visualizer


def build_freehand_config(
    *,
    working_formula: str,
    wl_from: float,
    wl_to: float,
    n_wl: int,
    angle_deg: float,
    touched: Mapping[str, bool],
    target: Mapping[str, np.ndarray | None],
    wl_um: np.ndarray | None = None,
    view_domain: Mapping[str, dict[str, list[float]]] | None = None,
    edit_wl_indices: Mapping[str, set[int] | list[int] | None] | None = None,
    cost_scope: str | None = None,
    polarization: str = "UNPOLARIZED",
) -> Dict[str, Any]:
    fv = filmstack_visualizer
    wl_step = (float(wl_to) - float(wl_from)) / max(int(n_wl) - 1, 1)
    scope = cost_scope if cost_scope is not None else get_freehand_cost_scope()
    runtime: Dict[str, Any] = {
        "formula": working_formula,
        "target_wl": [float(wl_from), float(wl_to), wl_step],
        "target_angle": [float(angle_deg), float(angle_deg)],
        "polarization": str(polarization).upper(),
        "freehand_touched": {k: bool(touched.get(k)) for k in ("R", "T", "A")},
        "freehand_cost_scope": scope,
    }
    if touched.get("R") and target.get("R") is not None:
        runtime["R_target_spectrum"] = np.asarray(target["R"], dtype=float).reshape(1, -1).tolist()
    if touched.get("T") and target.get("T") is not None:
        runtime["T_target_spectrum"] = np.asarray(target["T"], dtype=float).reshape(1, -1).tolist()
    if touched.get("A") and target.get("A") is not None:
        runtime["A_target_spectrum"] = np.asarray(target["A"], dtype=float).reshape(1, -1).tolist()
    if wl_um is not None:
        wl_indices = build_freehand_wl_indices(
            scope=scope,
            wl_um=np.asarray(wl_um, dtype=float),
            touched=touched,
            view_domain=view_domain or {},
            edit_wl_indices=edit_wl_indices or {},
        )
        if wl_indices:
            runtime["freehand_wl_indices"] = wl_indices
    return fv.merge_filmstack_optimization_config(load_freehand_base_config(), runtime)
