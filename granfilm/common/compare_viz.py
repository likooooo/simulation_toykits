"""DR comparison panels and case summary figures for GranFilm compare UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from granfilm.common.baseline import BaselineSpectrum
from granfilm.common.experiment import (
    chi2_reliance,
    interpolate_experiment_to_energy,
    load_experiment_dat,
    resolve_experiment_dat_path,
)
from granfilm.common.geometry_viz import plot_geometry_schematic
from granfilm.oblate_prolate.case import SpheroidCase
from granfilm.sphere_island.case import GranFilmCase

_PIPELINE = Literal["sphere", "spheroid"]


def _experiment_dr_on_grid(case: GranFilmCase | SpheroidCase, energy: np.ndarray) -> np.ndarray | None:
    if not case.compa or not case.expfilename.strip():
        return None
    exp_path = resolve_experiment_dat_path(case.path_dielectric, case.expfilename)
    if not exp_path.is_file():
        return None
    exp_spec = load_experiment_dat(exp_path)
    return interpolate_experiment_to_energy(exp_spec, energy)


def plot_dr_compare_panel(
    energy: np.ndarray,
    python_dr: np.ndarray,
    baseline: BaselineSpectrum,
    out: Path,
    *,
    title: str = "DR vs Fortran baseline",
    experiment_dr: np.ndarray | None = None,
    experiment_label: str | None = None,
) -> dict[str, float | str]:
    """Plot Python vs baseline DR (left axis) and ΔDR (right axis)."""
    py_on_base = np.interp(baseline.energy_ev, energy, python_dr)
    diff = py_on_base - baseline.value

    fig, ax_l = plt.subplots(figsize=(9, 3.2))
    ax_l.plot(energy, python_dr, "C1-", lw=1.5, label="Python")
    ax_l.plot(baseline.energy_ev, baseline.value, "C0--", lw=1.2, label="GranFilm baseline")
    if experiment_dr is not None:
        ax_l.plot(
            energy,
            experiment_dr,
            ":",
            color="C3",
            lw=1.2,
            label=experiment_label or "Experiment",
        )
    ax_l.set_xlabel("Energy (eV)")
    ax_l.set_ylabel("DR", color="C0")
    ax_l.tick_params(axis="y", labelcolor="C0")
    ax_l.grid(True, alpha=0.3)

    ax_r = ax_l.twinx()
    ax_r.plot(baseline.energy_ev, diff, "C2-", lw=1.0, alpha=0.85, label="ΔDR")
    ax_r.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax_r.set_ylabel("ΔDR (Python − baseline)", color="C2")
    ax_r.tick_params(axis="y", labelcolor="C2")

    ax_l.set_title(title, fontsize=9)
    lines_l, labels_l = ax_l.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax_l.legend(lines_l + lines_r, labels_l + labels_r, fontsize=7, loc="best")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)

    metrics: dict[str, float | str] = {
        "max_abs_err": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "python_peak_E": float(energy[int(np.argmax(python_dr))]),
        "baseline_peak_E": float(baseline.energy_ev[int(np.argmax(baseline.value))]),
        "note": "Python uses gf database (GranFilm SOPRA .nk + finite_size)",
    }
    if experiment_dr is not None:
        metrics["chi2_reliance"] = chi2_reliance(python_dr, experiment_dr)
    return metrics


def plot_case_summary(
    case: GranFilmCase | SpheroidCase,
    *,
    pipeline: _PIPELINE,
    energy: np.ndarray,
    python_dr: np.ndarray,
    baseline: BaselineSpectrum,
    out: Path,
    metrics_path: Path | None = None,
    title: str | None = None,
) -> dict[str, float | str]:
    """Composite figure: geometry schematic (top) + dual-axis DR panel (bottom)."""
    geom_label = getattr(case, "geometry", pipeline)
    panel_title = title or f"DR ({geom_label})"
    experiment_dr = _experiment_dr_on_grid(case, energy)
    exp_label = f"Experiment ({case.expfilename})" if experiment_dr is not None else None

    import tempfile

    with tempfile.TemporaryDirectory(prefix="gf_geom_") as tmp:
        geom_path = Path(tmp) / "geometry.png"
        plot_geometry_schematic(case, pipeline=pipeline, out=geom_path)
        geom_img = plt.imread(geom_path)

    fig = plt.figure(figsize=(9, 5.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.4], hspace=0.35)

    ax_geo = fig.add_subplot(gs[0])
    ax_geo.imshow(geom_img)
    ax_geo.axis("off")
    ax_geo.set_title("Geometry", fontsize=9)

    ax_l = fig.add_subplot(gs[1])
    py_on_base = np.interp(baseline.energy_ev, energy, python_dr)
    diff = py_on_base - baseline.value

    ax_l.plot(energy, python_dr, "C1-", lw=1.5, label="Python")
    ax_l.plot(baseline.energy_ev, baseline.value, "C0--", lw=1.2, label="GranFilm baseline")
    if experiment_dr is not None:
        ax_l.plot(energy, experiment_dr, ":", color="C3", lw=1.2, label=exp_label)
    ax_l.set_xlabel("Energy (eV)")
    ax_l.set_ylabel("DR", color="C0")
    ax_l.tick_params(axis="y", labelcolor="C0")
    ax_l.grid(True, alpha=0.3)
    ax_l.set_title(panel_title, fontsize=9)

    ax_r = ax_l.twinx()
    ax_r.plot(baseline.energy_ev, diff, "C2-", lw=1.0, alpha=0.85)
    ax_r.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax_r.set_ylabel("ΔDR", color="C2")
    ax_r.tick_params(axis="y", labelcolor="C2")

    lines_l, labels_l = ax_l.get_legend_handles_labels()
    ax_l.legend(lines_l, labels_l, fontsize=7, loc="best")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)

    metrics: dict[str, float | str] = {
        "max_abs_err": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "python_peak_E": float(energy[int(np.argmax(python_dr))]),
        "baseline_peak_E": float(baseline.energy_ev[int(np.argmax(baseline.value))]),
        "note": "Python uses gf database (GranFilm SOPRA .nk + finite_size)",
    }
    if experiment_dr is not None:
        metrics["chi2_reliance"] = chi2_reliance(python_dr, experiment_dr)
        metrics["experiment_file"] = str(
            resolve_experiment_dat_path(case.path_dielectric, case.expfilename)
        )
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def compare_metrics(python_y: np.ndarray, baseline_y: np.ndarray) -> dict[str, float]:
    a = np.asarray(python_y, dtype=float)
    b = np.asarray(baseline_y, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return {"rmse": float("nan"), "max_abs": float("nan"), "corr": float("nan")}
    d = a[mask] - b[mask]
    rmse = float(np.sqrt(np.mean(d * d)))
    max_abs = float(np.max(np.abs(d)))
    if a[mask].size < 2:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(a[mask], b[mask])[0, 1])
    return {"rmse": rmse, "max_abs": max_abs, "corr": corr}
