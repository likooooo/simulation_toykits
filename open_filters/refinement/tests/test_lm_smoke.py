"""LM smoke tests (no TMM)."""

from __future__ import annotations

import unittest

from moremath import Levenberg_Marquardt


class TestLevenbergMarquardtSmoke(unittest.TestCase):
    def test_quadratic_minimum(self) -> None:
        def f(a: list[float]) -> list[float]:
            return [a[0] ** 2 + a[1] ** 2]

        def df(a: list[float]) -> list[list[float]]:
            return [[2.0 * a[0]], [2.0 * a[1]]]

        lm = Levenberg_Marquardt.Levenberg_Marquardt(
            f, df, [3.0, 4.0], [0.0], [1.0]
        )
        lm.set_stop_criteria(min_gradient=1e-12, acceptable_chi_2=1e-12, min_chi_2_change=1e-9)
        lm.prepare()
        for _ in range(100):
            status = lm.iterate()
            if status != Levenberg_Marquardt.IMPROVING:
                break
        self.assertAlmostEqual(lm.a[0], 0.0, places=2)
        self.assertAlmostEqual(lm.a[1], 0.0, places=2)
        self.assertLess(lm.get_chi_2(), 1e-6)


if __name__ == "__main__":
    unittest.main()
