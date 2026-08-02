"""Small-grid n-k derivative map regression."""

from __future__ import annotations

import unittest

import numpy as np

from bootstrap_simulation import bootstrap_toykits_session, simulation_available
from plot_nk_derivative_maps import compute_nk_derivative_maps
from simulation_derivatives import Polarization


@unittest.skipUnless(simulation_available(), "simulation runtime missing")
class TestNkDerivativeMaps(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        bootstrap_toykits_session()

    def test_nk_derivative_map_small_grid(self) -> None:
        n_grid = np.linspace(2.0, 2.2, 8)
        k_grid = np.linspace(0.0, 0.02, 8)
        _, _, of_map, sim_map, diff_map = compute_nk_derivative_maps(
            layer_index=0,
            wl_nm=550.0,
            angle_deg=0.0,
            pol=Polarization.TE,
            n_grid=n_grid,
            k_grid=k_grid,
            quantity="R",
        )
        rel = np.abs(diff_map) / np.maximum(np.abs(sim_map), 1e-9)
        self.assertLess(float(np.nanmax(rel)), 0.05)


if __name__ == "__main__":
    unittest.main()
