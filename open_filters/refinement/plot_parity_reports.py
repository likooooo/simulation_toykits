#!/usr/bin/env python3
"""Run parity scenarios and write comparison PNGs (abeles baseline vs simulation)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_CROSSCHECK = Path(__file__).resolve().parents[1] / "crosscheck"
if str(_CROSSCHECK) not in sys.path:
    sys.path.insert(0, str(_CROSSCHECK))

from bootstrap_simulation import bootstrap_toykits_session, simulation_available  # noqa: E402
from stack_spec import load_default_materials_db  # noqa: E402

from refinement.fixtures import PARITY_DA_TOL_NM, PARITY_MAX_ITER  # noqa: E402
from refinement.parity import build_parity_scenarios, run_parity  # noqa: E402
from refinement.plot_parity import plot_parity_report, plot_parity_summary  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate parity comparison plots.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Directory for PNG output",
    )
    parser.add_argument("--show", action="store_true", help="Show figures interactively")
    args = parser.parse_args()

    if not simulation_available():
        print("error: simulation runtime missing (.simulation_toolkits/simulation.so)", file=sys.stderr)
        return 1

    bootstrap_toykits_session()
    materials_db = load_default_materials_db()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list = []
    for label, problem in build_parity_scenarios(materials_db):
        report = run_parity(
            problem,
            max_iter=PARITY_MAX_ITER,
            da_tol_nm=PARITY_DA_TOL_NM,
            chi2_rtol=1e-3,
            label=label,
        )
        png = plot_parity_report(
            report,
            problem,
            out_dir / f"parity_{label}.png",
            show=args.show,
        )
        print(f"wrote {png}  [{'PASS' if report.ok else 'FAIL'}]  {report.message}")
        summary_rows.append((label, report, problem))

    summary_path = plot_parity_summary(
        summary_rows,
        out_dir / "parity_summary.png",
        show=args.show,
    )
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
