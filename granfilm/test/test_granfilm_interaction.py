"""Unit tests for granfilm.common.interaction (RPT renorm, lattice sums)."""

from __future__ import annotations

import numpy as np
import pytest

from granfilm.common.constants import PI
from granfilm.common.interaction import integral_rpt, renorm_polarizability


class TestRenormPolarizability:
    def test_renorm_changes_alpha_simple_case(self):
        """Barrera RPT renorm must alter dipole polarizabilities for non-trivial coverage."""
        alpha = np.array([1.0 + 0.1j, 0.5 + 0.05j], dtype=np.complex128)
        alpha_before = alpha.copy()

        renorm_polarizability(
            0.3,
            alpha,
            eps_vacuum=1.0,
            eps_substrate=2.25 + 0.0j,
            coverage=0.45,
            R=10.0,
            Rapparent=10.0,
            levels=50,
            density=0.45 / (PI * 10.0**2),
            above=True,
            nint=64,
        )

        assert np.all(np.isfinite(alpha))
        assert not np.allclose(alpha, alpha_before, rtol=1e-5, atol=1e-12)
        rel_change = np.abs((alpha - alpha_before) / alpha_before)
        assert np.any(rel_change > 1e-4)

    def test_integral_rpt_finite(self):
        integrals = integral_rpt(d=0.2, lmax=5.0, rapp=10.0, nint=32)
        assert integrals.shape == (6,)
        assert np.all(np.isfinite(integrals))
        assert np.all(integrals > 0.0)
