"""Dual-run parity: abeles vs simulation LM refinement trajectories."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Sequence

from moremath import Levenberg_Marquardt

from refinement.backends.abeles import AbelesBackend
from refinement.backends.simulation import SimulationBackend
from refinement.problem import RefinementProblem
from refinement.session import RefinementHistoryEntry, RefinementSession
from refinement.target_spec import RefinementTargetSpec


@dataclass
class ParityStepDiff:
    iteration: int
    max_da_nm: float
    chi2_abeles: float
    chi2_simulation: float
    chi2_rel_diff: float
    status_abeles: int
    status_simulation: int


@dataclass
class ParityReport:
    ok: bool
    steps: list[ParityStepDiff] = field(default_factory=list)
    max_da_nm: float = 0.0
    max_chi2_rel_diff: float = 0.0
    message: str = ""
    label: str = ""
    hist_abeles: list[RefinementHistoryEntry] = field(default_factory=list)
    hist_simulation: list[RefinementHistoryEntry] = field(default_factory=list)

    def assert_ok(
        self,
        *,
        da_tol_nm: float = 1e-3,
        chi2_rtol: float = 1e-5,
    ) -> None:
        if not self.ok:
            raise AssertionError(self.message)


def _clone_problem(problem: RefinementProblem) -> RefinementProblem:
    return RefinementProblem(
        deepcopy(problem.spec),
        problem.materials_db,
        list(problem.targets),
        refine_layer_indices=list(problem.refine_layer_indices),
    )


def _compare_histories(
    hist_a: list,
    hist_s: list,
    *,
    da_tol_nm: float,
    chi2_rtol: float,
) -> tuple[bool, list[ParityStepDiff], float, float, str]:
    if len(hist_a) != len(hist_s):
        return (
            False,
            [],
            float("inf"),
            float("inf"),
            f"history length mismatch: abeles={len(hist_a)} simulation={len(hist_s)}",
        )

    diffs: list[ParityStepDiff] = []
    max_da = 0.0
    max_chi2_rel = 0.0

    for entry_a, entry_s in zip(hist_a, hist_s):
        da = max(
            abs(a - b)
            for a, b in zip(entry_a.thicknesses_nm, entry_s.thicknesses_nm)
        )
        max_da = max(max_da, da)
        chi2_diff = abs(entry_a.chi_2 - entry_s.chi_2)
        scale = max(abs(entry_a.chi_2), abs(entry_s.chi_2), 1e-12)
        chi2_rel = chi2_diff / scale
        chi2_abs_tol = max(chi2_rtol * scale, 1e-6)
        chi2_abs_ok = chi2_diff <= chi2_abs_tol
        max_chi2_rel = max(max_chi2_rel, chi2_rel if not chi2_abs_ok else 0.0)
        diffs.append(
            ParityStepDiff(
                iteration=entry_a.iteration,
                max_da_nm=da,
                chi2_abeles=entry_a.chi_2,
                chi2_simulation=entry_s.chi_2,
                chi2_rel_diff=chi2_rel,
                status_abeles=entry_a.status,
                status_simulation=entry_s.status,
            )
        )

    ok = max_da <= da_tol_nm
    for entry_a, entry_s in zip(hist_a, hist_s):
        chi2_diff = abs(entry_a.chi_2 - entry_s.chi_2)
        scale = max(abs(entry_a.chi_2), abs(entry_s.chi_2), 1e-12)
        if chi2_diff > max(chi2_rtol * scale, 1e-6):
            ok = False
            break
    msg = (
        f"max_da={max_da:.3e} nm (tol {da_tol_nm}), "
        f"max_chi2_rel={max_chi2_rel:.3e} (tol {chi2_rtol})"
    )
    if not ok:
        worst = max(diffs, key=lambda d: d.max_da_nm + d.chi2_rel_diff)
        msg += (
            f"; worst iter {worst.iteration}: "
            f"da={worst.max_da_nm:.3e} chi2_rel={worst.chi2_rel_diff:.3e}"
        )
    return ok, diffs, max_da, max_chi2_rel, msg


def run_parity(
    problem: RefinementProblem,
    a0: Sequence[float] | None = None,
    *,
    max_iter: int = 30,
    da_tol_nm: float = 1e-3,
    chi2_rtol: float = 1e-5,
    of_root: Any = None,
    label: str = "",
) -> ParityReport:
    """Run identical LM problems with abeles and simulation backends; compare trajectories."""
    a0_list = list(a0) if a0 is not None else problem.get_parameters()

    prob_a = _clone_problem(problem)
    prob_s = _clone_problem(problem)

    sess_a = RefinementSession(prob_a, AbelesBackend(of_root=of_root), a0_list)
    sess_s = RefinementSession(prob_s, SimulationBackend(), a0_list)

    sess_a.prepare()
    sess_s.prepare()

    improving = Levenberg_Marquardt.IMPROVING
    for _ in range(max_iter):
        if sess_a.status != improving and sess_s.status != improving:
            break
        sess_a.iterate()
        sess_s.iterate()

    ok, steps, max_da, max_chi2_rel, msg = _compare_histories(
        sess_a.history,
        sess_s.history,
        da_tol_nm=da_tol_nm,
        chi2_rtol=chi2_rtol,
    )
    return ParityReport(
        ok=ok,
        steps=steps,
        max_da_nm=max_da,
        max_chi2_rel_diff=max_chi2_rel,
        message=msg,
        label=label,
        hist_abeles=list(sess_a.history),
        hist_simulation=list(sess_s.history),
    )


def make_r_target_problem(
    spec: Any,
    materials_db: dict[str, Any],
    *,
    wl_nm: float = 550.0,
    target_r: float,
    sigma: float = 0.01,
    angle_deg: float = 0.0,
    polarization: str = "TE",
) -> RefinementProblem:
    """Build a single-point R target refinement problem."""
    target = RefinementTargetSpec(
        kind="R",
        wavelengths_nm=[float(wl_nm)],
        values=[float(target_r)],
        tolerances=[float(sigma)],
        angle_deg=float(angle_deg),
        polarization=polarization,  # type: ignore[arg-type]
    )
    return RefinementProblem(deepcopy(spec), materials_db, [target])


def build_parity_scenarios(materials_db) -> list[tuple[str, RefinementProblem]]:
    """Standard P1/P2/P3 parity scenarios shared by tests and plot reports."""
    from openfilters_derivatives import POL_TE, openfilters_rt_spectrum  # noqa: WPS433

    from refinement.fixtures import PARITY_REFINEMENT_STACK, PARITY_SPECTRUM_WLS_NM  # noqa: WPS433

    spec = deepcopy(PARITY_REFINEMENT_STACK)
    r0, _ = openfilters_rt_spectrum(spec, materials_db, [550.0], 0.0, POL_TE)
    target_r = max(0.05, min(0.95, float(r0[0]) * 0.9))

    p1 = make_r_target_problem(
        deepcopy(spec), materials_db, target_r=target_r, sigma=0.01, polarization="TE"
    )

    wls = list(PARITY_SPECTRUM_WLS_NM)
    r_vals, _ = openfilters_rt_spectrum(spec, materials_db, wls, 0.0, POL_TE)
    p2 = RefinementProblem(
        deepcopy(spec),
        materials_db,
        [
            RefinementTargetSpec(
                kind="R_spectrum",
                wavelengths_nm=wls,
                values=[float(v * 0.92) for v in r_vals],
                tolerances=[0.01] * len(wls),
                polarization="TE",
            )
        ],
    )

    _, t_vals = openfilters_rt_spectrum(spec, materials_db, [550.0], 0.0, POL_TE)
    p3 = RefinementProblem(
        deepcopy(spec),
        materials_db,
        [
            RefinementTargetSpec(
                kind="T",
                wavelengths_nm=[550.0],
                values=[max(0.01, float(t_vals[0]) * 1.05)],
                tolerances=[0.01],
                polarization="TE",
            )
        ],
    )

    return [
        ("P1_single_R", p1),
        ("P2_R_spectrum", p2),
        ("P3_single_T", p3),
    ]
