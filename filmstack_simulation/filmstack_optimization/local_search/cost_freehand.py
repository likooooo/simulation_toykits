"""Freehand target cost: R/T/A with per-metric touched mask."""

from __future__ import annotations

from typing import Any, List

import numpy as np

import filmstack_optimization_utils as fos


def _wl_index_map(targets) -> dict[float, int]:
    wls = np.array(sorted({float(t.wl_um) for t in targets}), dtype=float)
    return {float(w): i for i, w in enumerate(wls)}


def _allowed_index_sets(indices_map: dict[str, Any]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for metric in ("R", "T", "A"):
        raw = indices_map.get(metric)
        if raw is not None:
            out[metric] = {int(i) for i in raw}
    return out


def metric_active(
    metric: str,
    wl_um: float,
    wl_to_idx: dict[float, int],
    allowed_sets: dict[str, set[int]],
) -> bool:
    allowed = allowed_sets.get(metric)
    if allowed is None:
        return True
    idx = wl_to_idx.get(float(wl_um))
    return idx is not None and idx in allowed


def _target_contributes(
    i: int,
    tgt,
    *,
    touched: dict[str, bool],
    a_targets: List[float | None],
    wl_to_idx: dict[float, int],
    allowed_sets: dict[str, set[int]],
) -> bool:
    if touched.get("R") and tgt.R_target is not None and metric_active(
        "R", tgt.wl_um, wl_to_idx, allowed_sets
    ):
        return True
    if touched.get("T") and tgt.T_target is not None and metric_active(
        "T", tgt.wl_um, wl_to_idx, allowed_sets
    ):
        return True
    if (
        touched.get("A")
        and a_targets[i] is not None
        and metric_active("A", tgt.wl_um, wl_to_idx, allowed_sets)
    ):
        return True
    return False


def _a_targets_for_ctx(ctx) -> List[float | None]:
    cfg = getattr(ctx, "optimization_cfg", {}) or {}
    a_spec = cfg.get("A_target_spectrum")
    if a_spec is None:
        return [None] * len(ctx.targets)
    arr = np.asarray(a_spec, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != 1:
        raise ValueError("A_target_spectrum must be 1 x N for freehand")
    n = len(ctx.targets)
    if arr.shape[1] == n:
        return [float(arr[0, i]) for i in range(n)]
    wls = np.array(sorted({t.wl_um for t in ctx.targets}), dtype=float)
    if arr.shape[1] != len(wls):
        raise ValueError(
            f"A_target_spectrum width {arr.shape[1]} != target wl count {len(wls)}"
        )
    wl_to_a = {float(wls[i]): float(arr[0, i]) for i in range(len(wls))}
    return [wl_to_a.get(float(t.wl_um)) for t in ctx.targets]


def freehand_target(x: np.ndarray, ctx) -> tuple[float, np.ndarray]:
    """Merit over touched R/T/A targets; A gradients chain through R/T adjoints."""
    touched = getattr(ctx, "freehand_touched", None) or {}
    if not any(touched.get(k) for k in ("R", "T", "A")):
        raise ValueError("freehand_target requires at least one touched metric")

    layers_list, angles, wls = fos.build_layers_per_target(x, ctx)
    obs_list = fos.batch_rt_power_at_targets(layers_list, angles, wls)
    a_targets = _a_targets_for_ctx(ctx) if touched.get("A") else [None] * len(ctx.targets)
    wl_to_idx = _wl_index_map(ctx.targets)
    cfg = getattr(ctx, "optimization_cfg", {}) or {}
    scope = str(cfg.get("freehand_cost_scope", "full")).lower()
    if scope == "full":
        allowed_sets: dict[str, set[int]] = {}
    else:
        allowed_sets = _allowed_index_sets(getattr(ctx, "freehand_wl_indices", None) or {})

    contributing = [
        _target_contributes(
            i,
            tgt,
            touched=touched,
            a_targets=a_targets,
            wl_to_idx=wl_to_idx,
            allowed_sets=allowed_sets,
        )
        for i, tgt in enumerate(ctx.targets)
    ]
    weight_sum = sum(ctx.targets[i].weight for i, ok in enumerate(contributing) if ok) or 1.0

    total_loss = 0.0
    grad_per_target: List[List[float]] = [
        [0.0] * len(layers_list[0]) for _ in ctx.targets
    ]

    for i, tgt in enumerate(ctx.targets):
        if not contributing[i]:
            continue

        R, T = fos.combine_from_obs(obs_list[i], ctx.polarization)
        w_norm = tgt.weight / weight_sum
        loss_i = 0.0
        dL_dR = 0.0
        dL_dT = 0.0

        if (
            touched.get("R")
            and tgt.R_target is not None
            and metric_active("R", tgt.wl_um, wl_to_idx, allowed_sets)
        ):
            Rt = float(tgt.R_target)
            loss_i += (R - Rt) ** 2
            dL_dR += 2.0 * w_norm * (R - Rt)

        if (
            touched.get("T")
            and tgt.T_target is not None
            and metric_active("T", tgt.wl_um, wl_to_idx, allowed_sets)
        ):
            Tt = float(tgt.T_target)
            loss_i += (T - Tt) ** 2
            dL_dT += 2.0 * w_norm * (T - Tt)

        if (
            touched.get("A")
            and a_targets[i] is not None
            and metric_active("A", tgt.wl_um, wl_to_idx, allowed_sets)
        ):
            A = 1.0 - R - T
            At = float(a_targets[i])
            loss_i += (A - At) ** 2
            dA = 2.0 * w_norm * (A - At)
            dL_dR += -dA
            dL_dT += -dA

        total_loss += w_norm * loss_i

        if dL_dR != 0.0:
            dL_dR_list = [0.0] * len(ctx.targets)
            dL_dR_list[i] = dL_dR
            g_r = fos.batch_thickness_reflectance_adjoint_at_targets(
                layers_list, angles, wls, dL_dR_list, ctx.polarization
            )
            for j in range(len(grad_per_target[i])):
                grad_per_target[i][j] += g_r[i][j]

        if dL_dT != 0.0:
            dL_dT_list = [0.0] * len(ctx.targets)
            dL_dT_list[i] = dL_dT
            g_t = fos.batch_thickness_transmittance_adjoint_at_targets(
                layers_list, angles, wls, dL_dT_list, ctx.polarization
            )
            for j in range(len(grad_per_target[i])):
                grad_per_target[i][j] += g_t[i][j]

    grad = fos.accumulate_film_grad(grad_per_target, ctx.spec.film_indices)
    return float(total_loss), grad
