"""Visualization for oblate/prolate spheroid pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from granfilm.common.baseline import BaselineSpectrum, load_baseline
from granfilm.common.experiment import (
    chi2_reliance,
    interpolate_experiment_to_energy,
    load_experiment_dat,
    resolve_experiment_dat_path,
)
from granfilm.oblate_prolate.pipeline import SpheroidResult


def plot_step0_eps(result: SpheroidResult, out: Path) -> None:
    st = result.init
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(st.energy, np.real(st.eps_island), label="Re ε island")
    ax.plot(st.energy, np.imag(st.eps_island), label="Im ε island")
    ax.plot(st.energy, np.real(st.eps_substrate), "--", label="Re ε substrate")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("ε")
    ax.set_title(
        f"Step 0: {st.island_type} spheroid "
        f"(R_par={st.case.R_par}, R_per={st.case.R_per}, xi0={st.xi0:.4f})"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_step1_integrals(result: SpheroidResult, out: Path) -> None:
    mpo = result.init.case.Mpole_order
    qs = [result.integrals.Q[l, 1, 0] for l in range(1, mpo + 1)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, mpo + 1), qs, "o-", label="Q[l,1,0]")
    ax.axhline(result.integrals.Q[0, 0, 0], color="r", ls="--", label=f"Q000={result.integrals.Q[0,0,0]:.4f}")
    ax.set_xlabel("l")
    ax.set_title(f"Step 1: oblate/prolate integrals (tr={result.init.case.tr})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_step5_dr_compare(
    result: SpheroidResult,
    baseline: BaselineSpectrum,
    out: Path,
    metrics_path: Path | None = None,
) -> dict[str, float | str]:
    case = result.init.case
    experiment_on_grid: np.ndarray | None = None

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(result.energy, result.dr, label="Python oblate_prolate", lw=1.5)
    ax.plot(baseline.energy_ev, baseline.value, "--", label="GranFilm baseline", lw=1.2)

    if case.compa and case.expfilename.strip():
        exp_path = resolve_experiment_dat_path(case.path_dielectric, case.expfilename)
        if exp_path.is_file():
            exp_spec = load_experiment_dat(exp_path)
            experiment_on_grid = interpolate_experiment_to_energy(exp_spec, result.energy)
            ax.plot(
                result.energy,
                experiment_on_grid,
                ":",
                color="C3",
                label=f"Experiment ({case.expfilename})",
                lw=1.2,
            )

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("DR")
    ax.set_title(f"Step 5: DR ({result.init.island_type})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)

    py_on_base = np.interp(baseline.energy_ev, result.energy, result.dr)
    diff = py_on_base - baseline.value
    metrics: dict[str, float | str] = {
        "max_abs_err": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "python_peak_E": float(result.energy[int(np.argmax(result.dr))]),
        "baseline_peak_E": float(baseline.energy_ev[int(np.argmax(baseline.value))]),
        "note": f"oblate_prolate vs {baseline.source}",
    }
    if experiment_on_grid is not None:
        metrics["chi2_reliance"] = chi2_reliance(result.dr, experiment_on_grid)
        metrics["experiment_file"] = str(exp_path)
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def write_all_figures(
    result: SpheroidResult,
    viz_dir: Path,
    baseline: BaselineSpectrum | None = None,
) -> None:
    steps = viz_dir / "steps"
    steps.mkdir(parents=True, exist_ok=True)
    plot_step0_eps(result, steps / "step0_eps.png")
    plot_step1_integrals(result, steps / "step1_integrals.png")
    if baseline is not None:
        plot_step5_dr_compare(result, baseline, viz_dir / "compare.png", metrics_path=viz_dir / "metrics.json")
