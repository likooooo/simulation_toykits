#!/usr/bin/env python3
"""Binary-search alignment diagnostics for GranFilm pipeline steps."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from granfilm.paths import BISECT_LOG

LOG_PATH = BISECT_LOG


def _json_default(obj: object):
    if isinstance(obj, complex):
        return {"re": float(obj.real), "im": float(obj.imag)}
    return float(obj)


def _log(hypothesis_id: str, location: str, message: str, data: dict[str, Any], *, run_id: str = "bisect-pre") -> None:
    # #region agent log
    payload = {
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=_json_default) + "\n")
    # #endregion


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _integrals_max_diff(a: Any, b: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in ("Q", "K", "L", "M", "N"):
        va = getattr(a, name)
        vb = getattr(b, name)
        out[name] = _max_abs(va, vb)
    return out


def gauleg_fortran(x1: float, x2: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Port of legendre.f90 gauleg_dp (Numerical Recipes)."""
    from granfilm.common.legendre import arth

    eps = 3.0e-14
    x = np.empty(n, dtype=np.float64)
    w = np.empty(n, dtype=np.float64)
    m = (n + 1) // 2
    xm = 0.5 * (x2 + x1)
    xl = 0.5 * (x2 - x1)
    z = np.cos(np.pi * (arth(1.0, 1.0, m) - 0.25) / (n + 0.5))
    unfinished = np.ones(m, dtype=bool)
    for _its in range(10):
        if not np.any(unfinished):
            break
        p1 = np.ones(m, dtype=np.float64)
        p2 = np.zeros(m, dtype=np.float64)
        for j in range(1, n + 1):
            if not np.any(unfinished):
                break
            p3 = p2.copy()
            p2 = p1.copy()
            p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
        pp = n * (z * p1 - p2) / (z * z - 1.0)
        z1 = z.copy()
        z = z1 - p1 / pp
        unfinished = np.abs(z - z1) > eps
    x[:m] = xm - xl * z
    x[n - m :][::-1] = xm + xl * z
    w[:m] = 2.0 * xl / ((1.0 - z * z) * pp * pp)
    w[n - m :][::-1] = w[:m]
    return x, w


def step1_integrals_with_quadrature(
    tr: float,
    MPpos: float,
    mpole_order: int,
    m_max: int,
    *,
    nint: int,
    quadrature: str,
    fix_l0_ln: bool = False,
) -> Any:
    """step1_integrals variant for bisection (quadrature + optional L/N fix at l1=0)."""
    from granfilm.common.legendre import ass_legendre, deriv_ass_legendre, gauss_legendre
    from granfilm.sphere_island.step1_integrals import IntegralsVolume, _chi_func, _gamma_func, _mu_func

    mpo = mpole_order
    mm = m_max
    Int = IntegralsVolume.zeros(mpo, mm)

    if quadrature == "leggauss":
        x, w = gauss_legendre(-1.0, tr, nint)
    elif quadrature == "gauleg":
        x, w = gauleg_fortran(-1.0, tr, nint)
    else:
        raise ValueError(quadrature)

    mu = np.array([_mu_func(tr, 1, MPpos), _mu_func(tr, 2, MPpos)], dtype=np.float64)
    gamma = np.stack([_gamma_func(x, mu[0]), _gamma_func(x, mu[1])], axis=0)
    chi = np.stack([_chi_func(x, mu[0]), _chi_func(x, mu[1])], axis=0)

    poly_p: dict[tuple[int, int], np.ndarray] = {}
    poly_pp: dict[tuple[int, int, int], np.ndarray] = {}
    poly_dp: dict[tuple[int, int, int], np.ndarray] = {}

    poly_p[(0, 0)] = np.ones_like(x)
    for imu in range(2):
        poly_pp[(0, 0, imu)] = np.zeros_like(chi[imu])
        poly_dp[(0, 0, imu)] = np.zeros_like(chi[imu])
    for l in range(1, mpo + 1):
        for m in range(0, 2):
            poly_p[(l, m)] = ass_legendre(l, m, x)
            for imu in range(2):
                poly_pp[(l, m, imu)] = ass_legendre(l, m, chi[imu])
                poly_dp[(l, m, imu)] = deriv_ass_legendre(l, m, chi[imu])

    Int.Q[0, 0, 0] = tr + 1.0

    for l2 in range(mpo + 1):
        Int.Q[0, l2, 0] = float(np.sum(w * poly_p[(l2, 0)]))
        for imu in range(2):
            g = gamma[imu]
            Int.K[0, l2, 0, imu] = float(np.sum(w * poly_pp[(l2, 0, imu)] * g ** (-(l2 + 1) / 2.0)))
            l_core = (
                -(l2 + 1) * g ** (-0.5 * l2 - 1.5) * (1.0 - mu[imu] * x) * poly_pp[(l2, 0, imu)]
                + g ** (-0.5 * l2 - 2.0)
                * (x * g - (x - mu[imu]) * (1.0 - mu[imu] * x))
                * poly_dp[(l2, 0, imu)]
            )
            n_core = (
                l2 * g ** (0.5 * l2 - 1.0) * (1.0 - mu[imu] * x) * poly_pp[(l2, 0, imu)]
                + g ** (0.5 * l2 - 1.5)
                * (x * g - (x - mu[imu]) * (1.0 - mu[imu] * x))
                * poly_dp[(l2, 0, imu)]
            )
            if fix_l0_ln:
                Int.L[0, l2, 0, imu] = float(np.sum(w * l_core))
                Int.N[0, l2, 0, imu] = float(np.sum(w * n_core))
            else:
                Int.L[0, l2, 0, imu] = float(np.sum(w * poly_p[(l2, 0)] * l_core))
                Int.N[0, l2, 0, imu] = float(np.sum(w * poly_p[(l2, 0)] * n_core))
            Int.M[0, l2, 0, imu] = float(np.sum(w * poly_pp[(l2, 0, imu)] * g ** (l2 / 2.0)))

    for l1 in range(1, mpo + 1):
        Int.Q[l1, 0, 0] = float(np.sum(w * poly_p[(l1, 0)]))
        for imu in range(2):
            g = gamma[imu]
            Int.K[l1, 0, 0, imu] = float(np.sum(w * poly_p[(l1, 0)] * g ** (-0.5)))
            Int.L[l1, 0, 0, imu] = float(
                np.sum(
                    w
                    * poly_p[(l1, 0)]
                    * (
                        -g ** (-1.5) * (1.0 - mu[imu] * x)
                        + g ** (-2.0) * (x * g - (x - mu[imu]) * (1.0 - mu[imu] * x))
                    )
                )
            )
            Int.M[l1, 0, 0, imu] = float(np.sum(w * poly_p[(l1, 0)]))
            Int.N[l1, 0, 0, imu] = float(
                np.sum(
                    w
                    * poly_p[(l1, 0)]
                    * (g ** (-1.5) * (x * g - (x - mu[imu]) * (1.0 - mu[imu] * x)))
                )
            )

    for l1 in range(1, mpo + 1):
        for l2 in range(1, mpo + 1):
            for m in range(0, 2):
                Int.Q[l1, l2, m] = float(np.sum(w * poly_p[(l1, m)] * poly_p[(l2, m)]))
                for imu in range(2):
                    g = gamma[imu]
                    Int.K[l1, l2, m, imu] = float(
                        np.sum(w * poly_p[(l1, m)] * poly_pp[(l2, m, imu)] * g ** (-(l2 + 1) / 2.0))
                    )
                    Int.L[l1, l2, m, imu] = float(
                        np.sum(
                            w
                            * poly_p[(l1, m)]
                            * (
                                -(l2 + 1)
                                * g ** (-0.5 * l2 - 1.5)
                                * (1.0 - mu[imu] * x)
                                * poly_pp[(l2, m, imu)]
                                + g ** (-0.5 * l2 - 2.0)
                                * (x * g - (x - mu[imu]) * (1.0 - mu[imu] * x))
                                * poly_dp[(l2, m, imu)]
                            )
                        )
                    )
                    Int.M[l1, l2, m, imu] = float(
                        np.sum(w * poly_p[(l1, m)] * poly_pp[(l2, m, imu)] * g ** (l2 / 2.0))
                    )
                    Int.N[l1, l2, m, imu] = float(
                        np.sum(
                            w
                            * poly_p[(l1, m)]
                            * (
                                l2 * g ** (0.5 * l2 - 1.0) * (1.0 - mu[imu] * x) * poly_pp[(l2, m, imu)]
                                + g ** (0.5 * l2 - 1.5)
                                * (x * g - (x - mu[imu]) * (1.0 - mu[imu] * x))
                                * poly_dp[(l2, m, imu)]
                            )
                        )
                    )
    return Int


def run_bisect(*, run_id: str = "bisect-pre") -> None:
    from granfilm.common.baseline import default_baseline_path, load_baseline
    from granfilm.sphere_island.case import default_sphere_case
    from granfilm.common.materials import build_granfilm_materials_db
    from granfilm.sphere_island.pipeline import run_granfilm_sphere
    from granfilm.common.sopra_dielectric import reference_from_granfilm_tree
    from granfilm.sphere_island.step0_init import step0_init
    from granfilm.sphere_island.step1_integrals import step1_integrals
    from granfilm.common.zeta import step1_zeta

    case = default_sphere_case()
    materials_db = build_granfilm_materials_db()
    baseline = load_baseline(default_baseline_path())
    state = step0_init(case, materials_db)
    energy = state.energy
    mid = len(energy) // 2

    gf_dir = Path("/home/like/repos/GranFilm-v1.0")
    ref_eps_ag = reference_from_granfilm_tree(
        gf_dir / "Dielectric",
        "ag",
        energy,
        geometry=case.geometry,
        tr=case.tr,
        R_nm=case.R,
        mean_free_path=case.mean_free_path,
        surface_effects=case.surface_effects,
    )
    eps_err = float(np.max(np.abs(state.eps_island - ref_eps_ag)))
    _log("A", "bisect:step0", "step0 eps_island vs GranFilm .nk", {"max_abs_err": eps_err, "aligned_1e5": eps_err < 1e-5}, run_id=run_id)

    zeta = step1_zeta(case.Mpole_order, state.m_max)
    zeta_ref = step1_zeta(case.Mpole_order, state.m_max)
    zeta_err = _max_abs(zeta, zeta_ref)
    _log("B", "bisect:step1a", "zeta self-consistency", {"max_abs_err": zeta_err, "aligned_1e5": zeta_err < 1e-5}, run_id=run_id)

    x_lg, w_lg = __import__("granfilm.legendre", fromlist=["gauss_legendre"]).gauss_legendre(-1.0, case.tr, case.Nint)
    x_gl, w_gl = gauleg_fortran(-1.0, case.tr, case.Nint)
    quad_err = max(_max_abs(x_lg, x_gl), _max_abs(w_lg, w_gl))
    _log("C", "bisect:step1b-quad", "leggauss vs fortran gauleg", {"max_abs_err": quad_err, "aligned_1e5": quad_err < 1e-5}, run_id=run_id)

    int_cur = step1_integrals(case.tr, case.MPpos, case.Mpole_order, state.m_max, nint=case.Nint)
    int_ref = step1_integrals_with_quadrature(
        case.tr, case.MPpos, case.Mpole_order, state.m_max, nint=case.Nint, quadrature="gauleg", fix_l0_ln=True
    )
    int_errs = _integrals_max_diff(int_cur, int_ref)
    _log(
        "D",
        "bisect:step1b-int",
        "current integrals vs gauleg+fortran L/N(l1=0)",
        {**int_errs, "max_all": max(int_errs.values()), "aligned_1e5": max(int_errs.values()) < 1e-5},
        run_id=run_id,
    )

    peak_i = int(np.argmax(baseline.value))
    _log(
        "F",
        "bisect:peak-index",
        "baseline DR peak location",
        {"peak_i": peak_i, "peak_E_eV": float(baseline.energy_ev[peak_i]), "baseline_peak_DR": float(baseline.value[peak_i])},
        run_id=run_id,
    )

    from granfilm.sphere_island.step2_system import _matrix_system_above, _right_dipole_above
    from granfilm.common.linsolver import linsolve_granfilm, _relative_residual

    z = zeta
    int_tr1 = step1_integrals(1.0, case.MPpos, case.Mpole_order, state.m_max, nint=case.Nint)
    for label, i_e in [("mid", mid), ("peak", peak_i)]:
        for m in (0, 1):
            A = _matrix_system_above(m, i_e, state, int_cur, int_tr1, z)
            b = _right_dipole_above(m, i_e, state, int_cur, z)
            x_naive = np.linalg.solve(A, b)
            x_ref = linsolve_granfilm(A, b, epslin=case.epslin)
            _log(
                "F",
                f"bisect:step2-{label}-m{m}",
                "step2 solver naive vs refined",
                {
                    "energy_eV": float(energy[i_e]),
                    "m": m,
                    "residual_naive": _relative_residual(A, x_naive, b),
                    "residual_refined": _relative_residual(A, x_ref, b),
                    "x_max_diff": float(np.max(np.abs(x_naive - x_ref))),
                    "x0_naive": complex(x_naive[0]),
                    "x0_refined": complex(x_ref[0]),
                },
                run_id=run_id,
            )

    int_l0_only = {
        "L_0_l2_0_imu0": abs(int_cur.L[0, 2, 0, 0] - int_ref.L[0, 2, 0, 0]),
        "L_0_l2_0_imu1": abs(int_cur.L[0, 2, 0, 1] - int_ref.L[0, 2, 0, 1]),
        "N_0_l2_0_imu0": abs(int_cur.N[0, 2, 0, 0] - int_ref.N[0, 2, 0, 0]),
    }
    _log("D", "bisect:step1b-l0", "L/N at (l1=0,l2=2) sample", int_l0_only, run_id=run_id)

    from granfilm.sphere_island.step2_system import step2_solve_multipoles
    from granfilm.sphere_island.step3_polarizability import step3_polarizabilities
    from granfilm.sphere_island.step4_interaction import step4_surface_coefficients
    from granfilm.sphere_island.step5_fresnel import step5_fresnel

    z = zeta
    int_tr1 = step1_integrals(1.0, case.MPpos, case.Mpole_order, state.m_max, nint=case.Nint)
    for label, i_e in [("mid", mid), ("peak", peak_i)]:
        mpo, _mpq = step2_solve_multipoles(state, int_cur, int_tr1, z, i_e)
        alpha = step3_polarizabilities(mpo, state, i_e)
        chi = step4_surface_coefficients(alpha, state, i_e, nint=case.Nint)
        dr_i = step5_fresnel(chi, state, i_e)
        from granfilm.sphere_island.step4_interaction import lattice_sum

        _log(
            "G",
            f"bisect:pipeline-{label}",
            "step2-5 checkpoint vs baseline DR",
            {
                "energy_eV": float(energy[i_e]),
                "dr_python": float(dr_i),
                "dr_baseline": float(baseline.value[i_e]),
                "dr_abs_err": float(abs(dr_i - baseline.value[i_e])),
                "mpo_00": complex(mpo[0, 0]),
                "mpo_01": complex(mpo[0, 1]),
                "alpha_00": complex(alpha[0, 0]),
                "alpha_01": complex(alpha[0, 1]),
                "S_mp": float(lattice_sum(0.0, 2, state, nint=case.Nint)),
                "S_imp": float(lattice_sum(abs(case.tr - case.MPpos), 2, state, nint=case.Nint)),
                "chi_gamma": complex(chi.gamma),
                "chi_beta": complex(chi.beta),
            },
            run_id=run_id,
        )

    result = run_granfilm_sphere(case, materials_db, write_viz=False)
    dr_err = float(np.max(np.abs(result.dr - baseline.value)))
    dr_rmse = float(np.sqrt(np.mean((result.dr - baseline.value) ** 2)))
    _log(
        "E",
        "bisect:step5",
        "final DR vs SphereTest.dat baseline",
        {"max_abs_err": dr_err, "rmse": dr_rmse, "aligned_1e5": dr_err < 1e-5},
        run_id=run_id,
    )


if __name__ == "__main__":
    import sys

    run_id = sys.argv[1] if len(sys.argv) > 1 else "bisect-pre"
    run_bisect(run_id=run_id)
