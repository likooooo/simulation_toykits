"""Phase 2: thickness derivative cross-check."""

from __future__ import annotations

import unittest

from bootstrap_simulation import bootstrap_toykits_session, simulation_available
from compare import compare_thickness_derivatives
from refinement.fixtures import PARITY_COMPLEX_STACK, PARITY_REFINEMENT_STACK
from simulation_derivatives import Polarization
from stack_spec import DEFAULT_BRAGG_STACK, load_default_materials_db


@unittest.skipUnless(simulation_available(), "simulation runtime missing")
class TestThicknessDerivatives(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        bootstrap_toykits_session()
        cls.materials_db = load_default_materials_db()

    def _assert_compare(self, angle_deg: float, pol: str, wls: list[float]) -> None:
        result = compare_thickness_derivatives(
            DEFAULT_BRAGG_STACK, self.materials_db, wls, angle_deg, pol, rtol=5e-3
        )
        self.assertTrue(result.ok_rt, result.message)
        self.assertTrue(result.ok_dR, result.message)
        self.assertTrue(result.ok_dT, result.message)

    def test_normal_te(self) -> None:
        self._assert_compare(0.0, Polarization.TE, [550.0])

    def test_normal_tm(self) -> None:
        self._assert_compare(0.0, Polarization.TM, [550.0])

    def test_oblique_te(self) -> None:
        self._assert_compare(60.0, Polarization.TE, [550.0])

    def test_oblique_unpolarized(self) -> None:
        self._assert_compare(60.0, Polarization.UNPOLARIZED, [550.0])

    def test_multi_wavelength(self) -> None:
        self._assert_compare(0.0, Polarization.TE, [400.0, 550.0, 700.0])

    def test_parity_refinement_stack(self) -> None:
        """Gate stack for LM parity: both backends must agree on df."""
        result = compare_thickness_derivatives(
            PARITY_REFINEMENT_STACK,
            self.materials_db,
            [550.0],
            0.0,
            Polarization.TE,
            rtol=5e-3,
        )
        self.assertTrue(result.ok_rt, result.message)
        self.assertTrue(result.ok_dR, result.message)
        self.assertTrue(result.ok_dT, result.message)

    def test_complex_stack_derivatives(self) -> None:
        """Simulation C++ adjoint vs abeles for >=3 films (forward R/T OK)."""
        result = compare_thickness_derivatives(
            PARITY_COMPLEX_STACK,
            self.materials_db,
            [550.0],
            0.0,
            Polarization.TE,
            rtol=5e-3,
        )
        self.assertTrue(result.ok_dR, result.message)
        self.assertTrue(result.ok_dT, result.message)


if __name__ == "__main__":
    unittest.main()
