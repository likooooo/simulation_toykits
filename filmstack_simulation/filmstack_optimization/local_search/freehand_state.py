"""Session state for Freehand R/T/A editing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from filmstack_simulation.materials import RECOMMENDED_SIM_WL_FROM_UM, RECOMMENDED_SIM_WL_TO_UM

METRICS = ("R", "T", "A")
Y_PERCENT_CAP = 100.0
Y_PERCENT_FLOOR = 1.0


def auto_y_max_percent(*series: np.ndarray | None) -> float:
    max_val = max(
        (float(np.max(s)) for s in series if s is not None and s.size),
        default=0.0,
    )
    return max(min(max_val * 100.0 * 1.2, Y_PERCENT_CAP), Y_PERCENT_FLOOR)


def auto_y_min_percent(*series: np.ndarray | None) -> float:
    mins = [float(np.min(s)) for s in series if s is not None and s.size]
    if not mins:
        return 0.0
    return max(0.0, min(mins) * 100.0 * 0.8)


def default_view_domain(
    wl_from: float,
    wl_to: float,
    current: Mapping[str, np.ndarray] | None = None,
    target: Mapping[str, np.ndarray | None] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        m: {
            "x": [float(wl_from), float(wl_to)],
            "y": [
                auto_y_min_percent(
                    current.get(m) if current else None,
                    target.get(m) if target else None,
                ),
                auto_y_max_percent(
                    current.get(m) if current else None,
                    target.get(m) if target else None,
                ),
            ],
            "yAuto": True,
        }
        for m in METRICS
    }


def wl_indices_in_x_range(wl_um: np.ndarray, x_lo: float, x_hi: float) -> list[int]:
    lo, hi = min(float(x_lo), float(x_hi)), max(float(x_lo), float(x_hi))
    eps = 1e-9
    return [i for i, w in enumerate(wl_um) if lo - eps <= float(w) <= hi + eps]


def build_freehand_wl_indices(
    *,
    scope: str,
    wl_um: np.ndarray,
    touched: Mapping[str, bool],
    view_domain: Mapping[str, dict[str, list[float]]],
    edit_wl_indices: Mapping[str, set[int] | list[int] | None],
) -> dict[str, list[int]] | None:
    if scope == "full":
        return None
    wl_arr = np.asarray(wl_um, dtype=float)
    if wl_arr.size == 0:
        return None
    out: dict[str, list[int]] = {}
    for m in METRICS:
        if not touched.get(m):
            continue
        if scope == "stroke":
            raw = edit_wl_indices.get(m) or []
            if raw:
                out[m] = sorted(int(i) for i in raw)
        elif scope == "zoom":
            dom = view_domain.get(m) or {}
            x = dom.get("x") or [float(wl_arr[0]), float(wl_arr[-1])]
            idx = wl_indices_in_x_range(wl_arr, x[0], x[1])
            if idx:
                out[m] = idx
    return out or None


@dataclass
class FreehandSession:
    working_formula: str = ""
    baseline_formula: str = ""
    last_optimized_formula: str | None = None
    opt_round: int = 0
    wl_um: np.ndarray = field(default_factory=lambda: np.array([]))
    angle_deg: float = 0.0
    polarization: str = "UNPOLARIZED"
    wl_from: float = RECOMMENDED_SIM_WL_FROM_UM
    wl_to: float = RECOMMENDED_SIM_WL_TO_UM
    current: dict[str, np.ndarray] = field(default_factory=dict)
    target: dict[str, np.ndarray | None] = field(default_factory=lambda: {m: None for m in METRICS})
    touched: dict[str, bool] = field(default_factory=lambda: {m: False for m in METRICS})
    active_metric: str = "R"
    last_merit_history: list[float] | None = None
    last_merit_initial: float | None = None
    view_domain: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    edit_wl_indices: dict[str, set[int]] = field(
        default_factory=lambda: {m: set() for m in METRICS}
    )
    built: bool = False
    optimizing: bool = False

    def clear_targets(self) -> None:
        for m in METRICS:
            self.target[m] = None
            self.touched[m] = False
            self.edit_wl_indices[m] = set()

    def reset_after_build(
        self,
        *,
        formula: str,
        wl_um: np.ndarray,
        angle_deg: float,
        current: dict[str, np.ndarray],
        wl_from: float,
        wl_to: float,
        polarization: str = "UNPOLARIZED",
    ) -> None:
        self.working_formula = formula
        self.baseline_formula = formula
        self.last_optimized_formula = None
        self.opt_round = 0
        self.wl_um = np.asarray(wl_um, dtype=float)
        self.angle_deg = float(angle_deg)
        self.polarization = str(polarization).upper()
        self.wl_from = float(wl_from)
        self.wl_to = float(wl_to)
        self.current = {k: np.asarray(v, dtype=float) for k, v in current.items()}
        self.clear_targets()
        self.last_merit_history = None
        self.last_merit_initial = None
        self.view_domain = default_view_domain(wl_from, wl_to, current=current)
        self.built = True
        self.optimizing = False

    def refresh_auto_y_domains(self) -> None:
        for m in METRICS:
            dom = self.view_domain.get(m)
            if dom is not None and dom.get("yAuto") is False:
                continue
            if dom is None:
                x = [float(self.wl_from), float(self.wl_to)]
            else:
                x = dom.get("x") or [float(self.wl_from), float(self.wl_to)]
            y_lo = auto_y_min_percent(self.current.get(m), self.target.get(m))
            y_hi = auto_y_max_percent(self.current.get(m), self.target.get(m))
            self.view_domain[m] = {
                "x": [float(x[0]), float(x[1])],
                "y": [y_lo, y_hi],
                "yAuto": True,
            }

    def apply_optimization_result(
        self,
        *,
        formula: str,
        current: dict[str, np.ndarray],
        merit_history: list[float],
        merit_initial: float | None = None,
    ) -> None:
        self.working_formula = formula
        self.last_optimized_formula = formula
        self.opt_round += 1
        self.current = {k: np.asarray(v, dtype=float) for k, v in current.items()}
        self.last_merit_history = list(merit_history)
        self.last_merit_initial = float(merit_initial) if merit_initial is not None else None
        self.clear_targets()
        self.refresh_auto_y_domains()
        self.optimizing = False

    def to_component_args(self) -> dict[str, Any]:
        return {
            "wl": self.wl_um.tolist(),
            "current": {m: self.current[m].tolist() for m in METRICS if m in self.current},
            "target": {
                m: (self.target[m].tolist() if self.target.get(m) is not None else None)
                for m in METRICS
            },
            "touched": dict(self.touched),
            "activeMetric": self.active_metric,
            "viewDomain": self.view_domain,
            "optimizing": self.optimizing,
            "meritHistory": self.last_merit_history,
        }
