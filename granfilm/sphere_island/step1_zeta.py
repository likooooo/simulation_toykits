"""Step 1a: zeta normalization coefficients."""

from __future__ import annotations

import numpy as np


def step1_zeta(mpole_order: int, m_max: int) -> np.ndarray:
    """
    Return zeta[l1, l2, m] for l1,l2 in 0..mpole_order, m in 0..m_max.
    Matches initialize_mod.f90 get_zeta.
    """
    mpo = mpole_order
    zeta = np.zeros((mpo + 1, mpo + 1, m_max + 1), dtype=np.float64)
    for m in range(m_max + 1):
        if m == 0:
            for l1 in range(mpo + 1):
                for l2 in range(mpo + 1):
                    zeta[l1, l2, m] = 0.5 * np.sqrt((2 * l1 + 1) * (2 * l2 + 1))
        elif m == 1:
            for l1 in range(1, mpo + 1):
                for l2 in range(1, mpo + 1):
                    factorial = 1.0 / (l1 * (l1 + 1) * l2 * (l2 + 1))
                    zeta[l1, l2, m] = 0.5 * np.sqrt((2 * l1 + 1) * (2 * l2 + 1) * factorial)
        elif m == 2:
            for l1 in range(2, mpo + 1):
                for l2 in range(2, mpo + 1):
                    factorial = 1.0 / (
                        (l1 - 1)
                        * l1
                        * (l1 + 1)
                        * (l1 + 2)
                        * (l2 - 1)
                        * l2
                        * (l2 + 1)
                        * (l2 + 2)
                    )
                    zeta[l1, l2, m] = 0.5 * np.sqrt((2 * l1 + 1) * (2 * l2 + 1) * factorial)
    return zeta
