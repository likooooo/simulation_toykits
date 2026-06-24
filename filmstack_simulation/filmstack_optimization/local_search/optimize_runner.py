"""Run L-BFGS-B for Freehand optimization."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

import filmstack_optimization_utils as fos
from filmstack_simulation.simulation import compute_rta_at_angle
from filmstack_simulation.filmstack_optimization.shared.stack_table import formula_from_stack


def run_freehand_optimize(
    cfg: Dict[str, Any],
    materials_db: Dict[str, Any],
) -> Tuple[str, list[float], Dict[str, np.ndarray], float]:
    """Return optimized formula, merit history, R/T/A curves, and initial merit."""
    spec = fos.stack_from_formula(cfg["formula"], materials_db)
    targets, target_wls, target_angles = fos.build_targets_from_cfg(cfg)
    pol = fos.Polarization(str(cfg["polarization"]).upper())
    ctx = fos.make_objective_context(spec, targets, pol, cfg)
    ctx.freehand_touched = dict(cfg.get("freehand_touched", {}))
    ctx.freehand_wl_indices = dict(cfg.get("freehand_wl_indices", {}))
    ctx.optimization_cfg = cfg

    cost_fn = fos.load_filmstack_cost_function(
        cfg["cost_function"]["path"], cfg["cost_function"]["name"]
    )
    x0 = np.array([spec.thicknesses_um[i] for i in spec.film_indices], dtype=float)
    merit_initial, _ = cost_fn(x0, ctx)
    opt_x, opt_result, merit_history, _ = fos._run_optimize(spec, ctx, cost_fn, cfg)

    thicknesses = list(spec.thicknesses_um)
    for idx, t in zip(spec.film_indices, opt_x):
        thicknesses[idx] = float(t)
    optimized_formula = formula_from_stack(spec.materials, thicknesses, materials_db)

    wls, angles, _, _ = fos.resolve_target_axes(cfg)
    wl_from, wl_to, n_wl = float(wls.min()), float(wls.max()), len(wls)
    angle_deg = float(angles[0])

    curves = compute_rta_at_angle(
        spec.materials,
        thicknesses,
        angle_deg,
        wl_from,
        wl_to,
        n_wl=n_wl,
        polarization=str(cfg.get("polarization", "UNPOLARIZED")),
    )
    current = {"R": curves["R"], "T": curves["T"], "A": curves["A"]}
    return optimized_formula, merit_history, current, float(merit_initial)
