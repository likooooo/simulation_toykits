"""Matplotlib visualizations for each GranFilm pipeline step."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

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
from granfilm.sphere_island.pipeline import GranFilmResult


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_step0_eps(result: GranFilmResult, out: Path) -> None:
    st = result.init
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(st.energy, np.real(st.eps_island), label="Re ε island (ag, sim_db)")
    ax.plot(st.energy, np.imag(st.eps_island), label="Im ε island")
    ax.plot(st.energy, np.real(st.eps_substrate), "--", label="Re ε substrate (mgo)")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("ε")
    ax.set_title(
        f"Step 0: dielectric constants\n"
        f"geometry={result.geometry}, R={st.case.R} nm, t1={st.case.film_thickness1} nm"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_step1a_zeta(result: GranFilmResult, out: Path) -> None:
    if result.zeta is None:
        return
    z0 = result.zeta[:, :, 0]
    z1 = result.zeta[1:, 1:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(z0, origin="lower", aspect="auto")
    axes[0].set_title("ζ(l₁,l₂, m=0)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(z1, origin="lower", aspect="auto")
    axes[1].set_title("ζ(l₁,l₂, m=1), l≥1")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.suptitle("Step 1a: zeta normalization")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_step1b_integrals(result: GranFilmResult, out: Path) -> None:
    if result.integrals is None:
        return
    mpo = result.init.case.Mpole_order
    ks = [result.integrals.K[l, 1, 0, 0] for l in range(1, mpo + 1)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, mpo + 1), ks, "o-", label="Int.K[l,1,0].mu(1)")
    ax.axhline(result.integrals.Q[0, 0, 0], color="r", ls="--", label=f"Q₀₀₀=tr+1={result.integrals.Q[0,0,0]:.3f}")
    ax.set_xlabel("l")
    ax.set_title(f"Step 1b: geometry integrals (tr={result.init.case.tr}, m=0)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_step2_matrix(A: np.ndarray, out: Path, *, energy_ev: float) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(np.log10(np.abs(A) + 1e-30), origin="lower", aspect="auto")
    ax.set_title(f"Step 2: |A| matrix (log10, m=0, E={energy_ev:.2f} eV)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_step2_mpoles(result: GranFilmResult, out: Path) -> None:
    if result.mpoled is None:
        return
    i = result.mid_energy_index
    mpo = np.abs(result.mpoled[i])
    fig, ax = plt.subplots(figsize=(5, 3))
    labels = ["l=1,m=0", "l=1,m=1", "l=2,m=0", "l=2,m=1"]
    ax.bar(labels, [mpo[0, 0], mpo[0, 1], mpo[1, 0], mpo[1, 1]])
    ax.set_ylabel("|multipole coeff|")
    ax.set_title(f"Step 2: multipole coeffs @ E={result.energy[i]:.2f} eV")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_step3_alpha(result: GranFilmResult, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result.energy, np.real(result.alpha[:, 0, 0]), label="Re α∥ (dipole)")
    ax.plot(result.energy, np.real(result.alpha[:, 0, 1]), label="Re α⊥ (dipole)")
    ax.set_xlabel("Energy (eV)")
    ax.set_title("Step 3: polarizability alpha")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_step4_chi(result: GranFilmResult, out: Path) -> None:
    g = np.array([c.gamma for c in result.chi])
    b = np.array([c.beta for c in result.chi])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result.energy, np.real(g), label="Re γ")
    ax.plot(result.energy, np.real(b), label="Re β")
    ax.set_xlabel("Energy (eV)")
    ax.set_title("Step 4: surface coefficients gamma, beta (dipole+MFT)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_step5_dr_compare(
    result: GranFilmResult,
    baseline: BaselineSpectrum,
    out: Path,
    metrics_path: Path | None = None,
) -> dict[str, float | str]:
    case = result.init.case
    experiment_on_grid: np.ndarray | None = None

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(result.energy, result.dr, label="Python (sim_db ε)", lw=1.5)
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
    ax.set_title(f"Step 5: DR ({result.geometry})")
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
        "note": "Python uses gf database (GranFilm SOPRA .nk + finite_size)",
    }
    if experiment_on_grid is not None:
        metrics["chi2_reliance"] = chi2_reliance(result.dr, experiment_on_grid)
        metrics["experiment_file"] = str(
            resolve_experiment_dat_path(case.path_dielectric, case.expfilename)
        )
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def plot_step5_dr_only(result: GranFilmResult, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(result.energy, result.dr, lw=1.5)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("DR")
    ax.set_title(f"Step 5: DR ({result.geometry}, no Fortran baseline)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def write_all_step_figures(
    result: GranFilmResult,
    viz_dir: Path,
    *,
    matrix_fn: Callable[[], np.ndarray] | None = None,
    baseline: BaselineSpectrum | None = None,
) -> None:
    steps = viz_dir / "steps"
    _ensure_dir(steps)
    plot_step0_eps(result, steps / "step0_eps.png")
    plot_step1a_zeta(result, steps / "step1a_zeta.png")
    plot_step1b_integrals(result, steps / "step1b_integrals.png")
    if matrix_fn is not None:
        plot_step2_matrix(
            matrix_fn(),
            steps / "step2_matrix.png",
            energy_ev=float(result.energy[result.mid_energy_index]),
        )
    plot_step2_mpoles(result, steps / "step2_mpoles.png")
    plot_step3_alpha(result, steps / "step3_alpha.png")
    plot_step4_chi(result, steps / "step4_chi.png")

    if baseline is not None:
        plot_step5_dr_compare(
            result,
            baseline,
            viz_dir / "compare.png",
            metrics_path=viz_dir / "metrics.json",
        )
    elif result.geometry == "island":
        try:
            baseline = load_baseline()
        except (FileNotFoundError, ValueError):
            baseline = None
        if baseline is not None:
            plot_step5_dr_compare(
                result,
                baseline,
                viz_dir / "compare.png",
                metrics_path=viz_dir / "metrics.json",
            )
        else:
            plot_step5_dr_only(result, viz_dir / "compare.png")
    else:
        plot_step5_dr_only(result, viz_dir / "compare.png")
