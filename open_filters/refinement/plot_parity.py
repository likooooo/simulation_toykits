"""Plot abeles (baseline) vs simulation LM refinement parity."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from refinement.parity import ParityReport
from refinement.problem import RefinementProblem
from refinement.backends.abeles import AbelesBackend
from refinement.backends.simulation import SimulationBackend


def _layer_labels(problem: RefinementProblem) -> list[str]:
    labels: list[str] = []
    for layer_idx in problem.refine_layer_indices:
        tok = problem.spec.film_tokens[layer_idx]
        labels.append(f"L{layer_idx} {tok}")
    return labels


def _iterations(hist: Sequence) -> np.ndarray:
    return np.array([h.iteration for h in hist], dtype=int)


def plot_parity_report(
    report: ParityReport,
    problem: RefinementProblem,
    out_path: Path | str,
    *,
    show: bool = False,
) -> Path:
    """Save comparison figure: OpenFilters abeles (baseline) vs simulation."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ha = report.hist_abeles
    hs = report.hist_simulation
    if not ha or not hs:
        raise ValueError("ParityReport missing history; re-run run_parity()")

    n_layers = len(problem.refine_layer_indices)
    layer_names = _layer_labels(problem)
    iters = _iterations(ha)

    chi2_a = np.array([h.chi_2 for h in ha])
    chi2_s = np.array([h.chi_2 for h in hs])
    max_da = np.array([s.max_da_nm for s in report.steps], dtype=float)

    # Thickness trajectories per refined layer
    thick_a = np.array([h.thicknesses_nm for h in ha], dtype=float)
    thick_s = np.array([h.thicknesses_nm for h in hs], dtype=float)

    n_rows = 3 + (1 if _has_spectrum_residuals(problem) else 0)
    fig_h = 3.0 * n_rows
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, fig_h), constrained_layout=True)
    if n_rows == 1:
        axes = [axes]

    title = report.label or "LM refinement parity"
    status = "PASS" if report.ok else "FAIL"
    fig.suptitle(f"{title}  [{status}]  {report.message}", fontsize=11)

    # χ²
    ax = axes[0]
    ax.semilogy(iters, np.maximum(chi2_a, 1e-20), "o-", label="OpenFilters abeles (baseline)", color="C0")
    ax.semilogy(iters, np.maximum(chi2_s, 1e-20), "s--", label="simulation adjoint", color="C1")
    ax.set_xlabel("iteration")
    ax.set_ylabel("χ²")
    ax.set_title("Objective χ² vs iteration")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # max thickness delta
    ax = axes[1]
    if len(max_da) == len(iters[1:]):
        ax.plot(iters[1:], max_da, "o-", color="C2")
    else:
        ax.plot(range(len(max_da)), max_da, "o-", color="C2")
    ax.set_xlabel("iteration")
    ax.set_ylabel("max |Δd| (nm)")
    ax.set_title("Thickness mismatch (abeles − simulation)")
    ax.grid(True, alpha=0.3)

    # per-layer thickness (index by film layer in stack)
    ax = axes[2]
    for j, layer_idx in enumerate(problem.refine_layer_indices):
        ax.plot(iters, thick_a[:, layer_idx], "o-", label=f"{layer_names[j]} baseline")
        ax.plot(iters, thick_s[:, layer_idx], "s--", label=f"{layer_names[j]} simulation")
    ax.set_xlabel("iteration")
    ax.set_ylabel("thickness (nm)")
    ax.set_title("Layer thickness trajectories")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # spectrum / residual bar chart at final iterate
    if n_rows > 3:
        ax = axes[3]
        _plot_final_residuals(ax, problem, ha[-1], hs[-1])

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def _has_spectrum_residuals(problem: RefinementProblem) -> bool:
    for t in problem.targets:
        if t.kind in ("R_spectrum", "T_spectrum") and len(t.wavelengths_nm) > 1:
            return True
    return len(problem.all_target_values) > 1


def _plot_final_residuals(ax, problem: RefinementProblem, entry_a, entry_s) -> None:
    """Bar chart: target vs abeles vs simulation at final thicknesses."""
    from copy import deepcopy

    from refinement.problem import RefinementProblem

    a_final_a = [entry_a.thicknesses_nm[i] for i in problem.refine_layer_indices]
    a_final_s = [entry_s.thicknesses_nm[i] for i in problem.refine_layer_indices]

    prob_a = RefinementProblem(
        deepcopy(problem.spec),
        problem.materials_db,
        list(problem.targets),
        refine_layer_indices=list(problem.refine_layer_indices),
    )
    prob_a.apply_parameters(a_final_a)
    prob_s = RefinementProblem(
        deepcopy(problem.spec),
        problem.materials_db,
        list(problem.targets),
        refine_layer_indices=list(problem.refine_layer_indices),
    )
    prob_s.apply_parameters(a_final_s)

    ya = AbelesBackend().evaluate_values(prob_a)
    ys = SimulationBackend().evaluate_values(prob_s)
    yt = problem.all_target_values
    sig = problem.all_tolerances

    x = np.arange(len(yt))
    width = 0.25
    ax.bar(x - width, yt, width, label="target", color="C3", alpha=0.85)
    ax.bar(x, ya, width, label="abeles (baseline)", color="C0", alpha=0.85)
    ax.bar(x + width, ys, width, label="simulation", color="C1", alpha=0.85)

    for i, (ti, si) in enumerate(zip(yt, sig)):
        ax.plot([i - width, i - width], [ti - si, ti + si], color="C3", lw=2)

    labels: list[str] = []
    for sl in problem.target_slices:
        t = problem.targets[sl.target_idx]
        for k in range(sl.start, sl.end):
            local = k - sl.start
            wl = t.wavelengths_nm[local]
            labels.append(f"{t.kind}\n{wl:.0f}nm")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("value")
    ax.set_title("Final spectrum / residuals (target ± σ)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)


def plot_parity_summary(
    reports: Sequence[tuple[str, ParityReport, RefinementProblem]],
    out_path: Path | str,
    *,
    show: bool = False,
) -> Path:
    """Overview figure for multiple parity scenarios."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(reports)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.2 * n), constrained_layout=True)
    if n == 1:
        axes = np.array([axes])

    for row, (label, report, problem) in enumerate(reports):
        ha, hs = report.hist_abeles, report.hist_simulation
        iters = _iterations(ha)
        chi2_a = np.maximum([h.chi_2 for h in ha], 1e-20)
        chi2_s = np.maximum([h.chi_2 for h in hs], 1e-20)
        max_da = [s.max_da_nm for s in report.steps]

        ax = axes[row, 0]
        ax.semilogy(iters, chi2_a, "o-", label="baseline")
        ax.semilogy(iters, chi2_s, "s--", label="simulation")
        ax.set_ylabel("χ²")
        ax.set_title(f"{label}  [{'PASS' if report.ok else 'FAIL'}]")
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.legend(fontsize=8)

        ax = axes[row, 1]
        xi = iters[1:] if len(max_da) == len(iters) - 1 else range(len(max_da))
        ax.plot(xi, max_da, "o-", color="C2")
        ax.set_ylabel("max |Δd| (nm)")
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)

    fig.suptitle("OpenFilters abeles vs simulation — parity summary", fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path
