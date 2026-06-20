"""core.materials 测试用例"""

import pandas as pd
import numpy as np
import pytest
from core.materials import get_nk_at_wavelength, with_nk_columns


class _FakeMat:
    def __init__(self, n=1.46, k=0.0):
        self._n = n
        self._k = k

    def nk_at_wavelength_um(self, wl_um):
        return complex(self._n, self._k)


class TestGetNkAtWavelength:
    def test_vacuum(self):
        nk = get_nk_at_wavelength({}, "Vacuum", 0.532)
        assert nk == 1.0 + 0.0j

    def test_unknown_material(self):
        nk = get_nk_at_wavelength({}, "Unknown", 0.532)
        assert nk == 1.0 + 0.0j

    def test_from_database_material(self):
        materials_db = {"SiO2": _FakeMat(1.46, 0.0)}
        nk = get_nk_at_wavelength(materials_db, "SiO2", 0.5)
        assert nk == 1.46 + 0.0j


class TestWithNkColumns:
    def test_empty_df(self):
        df = pd.DataFrame()
        out = with_nk_columns(df, 0.532, lambda name: 1.5 + 0.0j)
        assert out.empty

    def test_fills_nk_from_func(self):
        df = pd.DataFrame([
            {"Material": "A", "Thickness (um)": 0.1},
            {"Material": "B", "Thickness (um)": 0.2},
        ])
        out = with_nk_columns(
            df, 0.532,
            lambda name: (1.46 if name == "A" else 2.1) + 0.0j,
        )
        assert list(out["n"]) == [1.46, 2.1]
        assert list(out["k"]) == [0.0, 0.0]

    def test_preserves_existing_nk(self):
        df = pd.DataFrame([
            {"Material": "A", "Thickness (um)": 0.1, "n": 1.5, "k": 0.01},
        ])
        out = with_nk_columns(df, 0.532, lambda name: 1.0 + 0.0j)
        assert list(out["n"]) == [1.5]
        assert list(out["k"]) == [0.01]

    def test_mixed_existing_and_computed(self):
        df = pd.DataFrame([
            {"Material": "A", "Thickness (um)": 0.1, "n": 1.5, "k": 0.01},
            {"Material": "B", "Thickness (um)": 0.2},
        ])
        out = with_nk_columns(
            df, 0.532,
            lambda name: (2.0 + 0.0j) if name == "B" else (1.0 + 0.0j),
        )
        assert out["n"].iloc[0] == 1.5 and out["k"].iloc[0] == 0.01
        assert out["n"].iloc[1] == 2.0 and out["k"].iloc[1] == 0.0
