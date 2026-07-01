"""Phase 1: nk alignment between OpenFilters abeles and simulation_database of/*."""

from __future__ import annotations

import unittest

from bootstrap_simulation import simulation_available
from material_alignment import compare_material_alignment


@unittest.skipUnless(simulation_available(), "simulation runtime missing")
class TestMaterialAlignment(unittest.TestCase):
    def test_of_materials_nk_alignment(self) -> None:
        failures = compare_material_alignment()
        self.assertFalse(failures, "nk mismatch:\n" + "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
