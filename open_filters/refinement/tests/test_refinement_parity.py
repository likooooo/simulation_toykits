"""Abeles vs simulation LM refinement parity tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_CROSSCHECK = Path(__file__).resolve().parents[2] / "crosscheck"
if str(_CROSSCHECK) not in sys.path:
    sys.path.insert(0, str(_CROSSCHECK))

from bootstrap_simulation import bootstrap_toykits_session, simulation_available  # noqa: E402
from stack_spec import load_default_materials_db  # noqa: E402

from refinement.fixtures import PARITY_DA_TOL_NM, PARITY_MAX_ITER  # noqa: E402
from refinement.parity import build_parity_scenarios, run_parity  # noqa: E402


@unittest.skipUnless(simulation_available(), "simulation runtime missing")
class TestRefinementParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        bootstrap_toykits_session()
        cls.materials_db = load_default_materials_db()

    def test_parity_scenarios(self) -> None:
        for label, problem in build_parity_scenarios(self.materials_db):
            with self.subTest(scenario=label):
                report = run_parity(
                    problem,
                    max_iter=PARITY_MAX_ITER,
                    da_tol_nm=PARITY_DA_TOL_NM,
                    chi2_rtol=1e-3,
                    label=label,
                )
                self.assertTrue(report.ok, f"{label}: {report.message}")


if __name__ == "__main__":
    unittest.main()
