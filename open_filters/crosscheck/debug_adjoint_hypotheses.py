#!/usr/bin/env python3
"""Runtime hypothesis tests for 6-layer thickness adjoint divergence."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

LOG_PATH = Path("/home/like/repos/simulation_toykits/.cursor/debug-f1408b.log")
SESSION = "f1408b"


def _log(hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "pre-fix") -> None:
    payload = {
        "sessionId": SESSION,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> int:
    import os
    run_id = os.environ.get("DEBUG_RUN_ID", "pre-fix")
    from bootstrap_simulation import bootstrap_toykits_session
    from openfilters_derivatives import POL_TE, openfilters_rt_spectrum, openfilters_thickness_rt_derivatives
    from refinement.fixtures import PARITY_COMPLEX_STACK, PARITY_REFINEMENT_STACK
    from simulation_derivatives import Polarization, simulation_thickness_rt_derivatives
    from stack_spec import StackSpec, load_default_materials_db

    bootstrap_toykits_session()
    db = load_default_materials_db()

    def layer_report(spec: StackSpec, label: str) -> None:
        n_films = len(spec.film_tokens)
        n_layers = n_films + 2
        n_local = n_layers - 1
        dR_of, _ = openfilters_thickness_rt_derivatives(spec, db, [550.0], 0.0, POL_TE)
        dR_sim, _ = simulation_thickness_rt_derivatives(spec, db, [550.0], 0.0, Polarization.TE)
        r0, _ = openfilters_rt_spectrum(spec, db, [550.0], 0.0, POL_TE)
        d0 = 1e-3
        rows = []
        for j in range(n_films):
            sim_i = 1 + j
            th_p = list(spec.film_thicknesses_nm)
            th_p[j] += d0
            spec_p = replace(spec, film_thicknesses_nm=th_p)
            r_p, _ = openfilters_rt_spectrum(spec_p, db, [550.0], 0.0, POL_TE)
            fd = (float(r_p[0]) - float(r0[0])) / d0
            of_d = float(dR_of[j, 0])
            sim_d = float(dR_sim[sim_i, 0]) / 1000.0
            sim_d_wrong_idx = float(dR_sim[j, 0]) / 1000.0 if j < dR_sim.shape[0] else 0.0
            suffix_len = max(0, n_local - 1 - sim_i)
            rows.append(
                {
                    "film_j": j,
                    "sim_layer_i": sim_i,
                    "suffix_len": suffix_len,
                    "fd": fd,
                    "abeles": of_d,
                    "sim_correct_idx": sim_d,
                    "sim_wrong_idx_j": sim_d_wrong_idx,
                    "rel_abeles_sim": abs(of_d - sim_d) / max(abs(of_d), 1e-12),
                    "rel_fd_abeles": abs(fd - of_d) / max(abs(of_d), 1e-12),
                    "rel_fd_sim": abs(fd - sim_d) / max(abs(sim_d), 1e-12),
                    "sign_match": (of_d * sim_d) >= 0 or abs(of_d) < 1e-15,
                }
            )
        _log("A", "debug_adjoint_hypotheses.py:layer_report", f"{label} suffix-order fingerprint", {"stack": label, "rows": rows}, run_id)
        _log("B", "debug_adjoint_hypotheses.py:layer_report", f"{label} float-precision check", {
            "stack": label,
            "last_two_max_rel": max(r["rel_abeles_sim"] for r in rows[-2:]) if n_films >= 2 else None,
            "inner_max_rel": max(r["rel_abeles_sim"] for r in rows[:-2]) if n_films > 2 else None,
        }, run_id)
        _log("C", "debug_adjoint_hypotheses.py:layer_report", f"{label} index-mapping check", {
            "stack": label,
            "rows": [
                {
                    "film_j": r["film_j"],
                    "rel_correct_idx": r["rel_abeles_sim"],
                    "rel_wrong_idx": abs(r["abeles"] - r["sim_wrong_idx_j"]) / max(abs(r["abeles"]), 1e-12),
                }
                for r in rows
            ],
        }, run_id)
        _log("D", "debug_adjoint_hypotheses.py:layer_report", f"{label} unit-scale check", {
            "stack": label,
            "rows": [
                {
                    "film_j": r["film_j"],
                    "ratio_sim_over_abeles": r["sim_correct_idx"] / r["abeles"] if abs(r["abeles"]) > 1e-15 else None,
                }
                for r in rows
            ],
        }, run_id)

    layer_report(PARITY_REFINEMENT_STACK, "2-layer")
    layer_report(PARITY_COMPLEX_STACK, "6-layer")

    # Hypothesis E: only layers with suffix_len <= 1 match abeles (pre-fix expectation)
    spec6 = PARITY_COMPLEX_STACK
    n_films = len(spec6.film_tokens)
    n_local = n_films + 2 - 1
    dR_of, _ = openfilters_thickness_rt_derivatives(spec6, db, [550.0], 0.0, POL_TE)
    dR_sim, _ = simulation_thickness_rt_derivatives(spec6, db, [550.0], 0.0, Polarization.TE)
    pattern = []
    for j in range(n_films):
        sim_i = 1 + j
        suffix_len = max(0, n_local - 1 - sim_i)
        of_d = float(dR_of[j, 0])
        sim_d = float(dR_sim[sim_i, 0]) / 1000.0
        rel = abs(of_d - sim_d) / max(abs(of_d), 1e-12)
        pattern.append({"film_j": j, "suffix_len": suffix_len, "rel_abeles_sim": rel, "ok_at_1e4": rel < 1e-4})
    _log("A", "debug_adjoint_hypotheses.py:pattern", "suffix_len vs derivative error", {"pattern": pattern}, run_id)
    _log("E", "debug_adjoint_hypotheses.py:pattern", "prefix/suffix structure fingerprint", {
        "rule": "suffix_len<=1 should match if only suffix order is wrong",
        "all_match_when_suffix_le_1": all(p["ok_at_1e4"] for p in pattern if p["suffix_len"] <= 1),
        "any_bad_when_suffix_gt_1": any(not p["ok_at_1e4"] for p in pattern if p["suffix_len"] > 1),
        "all_layers_ok_at_1e4": all(p["ok_at_1e4"] for p in pattern),
    }, run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
