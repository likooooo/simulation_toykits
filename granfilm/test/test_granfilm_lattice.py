"""Unit tests for square / hexagonal lattice sums (interaction_mod.f90)."""

from __future__ import annotations

import math

import pytest

from granfilm.common.constants import PI
from granfilm.common.interaction import (
    argument,
    lattice_sum,
    sr_hexagonal,
    sr_square,
)


def _ref_argument(l: int, d: float, r: float) -> float:
    """Inline port of interaction_mod.f90 Argument(l,d,r)."""
    cosi = d / r
    if l == 2:
        return (3 * cosi**2 - 1.0) / 2.0 / r**3
    if l == 3:
        return (5 * cosi**3 - 3 * cosi) / 2.0 / r**4
    if l == 4:
        return (35 * cosi**4 - 30 * cosi**2 + 3.0) / 8.0 / r**5
    raise ValueError(f"Argument l={l} not supported")


def _ref_sr_square(d: float, l: int, level: int) -> float:
    """Fortran-faithful Sr_square(d,l,level) loop."""
    sm = 0.0
    sd = 0.0
    s = 0.0
    for m in range(level, 0, -1):
        mf = float(m)
        sm += _ref_argument(l, d, math.sqrt(mf**2 + d**2))
        sd += _ref_argument(l, d, math.sqrt(2.0 * mf**2 + d**2))
        s = 0.0
        for n in range(m, -1, -1):
            nf = float(n)
            s += _ref_argument(l, d, math.sqrt(mf**2 + nf**2 + d**2))
    sr = 4.0 * (2.0 * s - sd - sm)
    return sr * math.sqrt((2 * l + 1) / (4 * PI))


def _ref_sr_hexagonal(d: float, l: int, level: int) -> float:
    """Fortran-faithful Sr_hexagonal(d,l,level) loop."""
    sd = 0.0
    s = 0.0
    for m in range(level, 0, -1):
        mf = float(m)
        sd += _ref_argument(l, d, math.sqrt(3.0 * mf**2 + d**2))
    for m in range(level, 0, -1):
        mf = float(m)
        for n in range(-m + 1, m):
            nf = float(n)
            s += _ref_argument(l, d, math.sqrt(mf**2 + nf**2 + mf * nf + d**2))
    sr = 4.0 * (sd + s)
    return sr * math.sqrt((2 * l + 1) / (4 * PI))


class TestArgument:
    @pytest.mark.parametrize("l", [2, 3, 4])
    @pytest.mark.parametrize("d,r", [(0.5, 1.2), (1.0, 2.5), (0.0, 3.0)])
    def test_matches_reference(self, l: int, d: float, r: float) -> None:
        assert argument(l, d, r) == pytest.approx(_ref_argument(l, d, r), rel=0, abs=1e-15)


class TestSrSquare:
    @pytest.mark.parametrize(
        "d,l,level",
        [
            (0.0, 2, 1),
            (0.5, 2, 5),
            (1.0, 3, 10),
            (0.2, 2, 50),
            (0.75, 4, 20),
        ],
    )
    def test_matches_fortran_reference(self, d: float, l: int, level: int) -> None:
        ref = _ref_sr_square(d, l, level)
        got = sr_square(d, l, level)
        assert got == pytest.approx(ref, rel=0, abs=1e-14)


class TestSrHexagonal:
    @pytest.mark.parametrize(
        "d,l,level",
        [
            (0.0, 2, 1),
            (0.5, 2, 5),
            (1.0, 3, 10),
            (0.2, 2, 50),
            (0.75, 4, 20),
        ],
    )
    def test_matches_fortran_reference(self, d: float, l: int, level: int) -> None:
        ref = _ref_sr_hexagonal(d, l, level)
        got = sr_hexagonal(d, l, level)
        assert got == pytest.approx(ref, rel=0, abs=1e-14)


class TestLatticeSumWrapper:
    @pytest.mark.parametrize("network", ["SQUARE", "HEXAGONAL"])
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_lattice_sum_uses_sr(self, network: str, n: int) -> None:
        d_mu = 0.3
        R = 5.0
        lattice_const = 13.2
        levels = 8
        d = 2.0 * d_mu * R / lattice_const
        if network == "SQUARE":
            expected = sr_square(d, n, levels)
        else:
            expected = sr_hexagonal(d, n, levels)
        got = lattice_sum(
            d_mu,
            n,
            network=network,
            R=R,
            Rapparent=R,
            density=0.45 / (PI * R**2),
            lattice_const=lattice_const,
            levels=levels,
        )
        assert got == pytest.approx(expected, rel=0, abs=1e-14)
