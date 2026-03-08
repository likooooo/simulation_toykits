"""core.materials 测试用例"""

import pandas as pd
import numpy as np
import pytest
from core.materials import get_nk_at_wavelength, with_nk_columns


def _load_nk_fake(shelf, book, page):
    """模拟 load_nk: 返回 (wls, ns, ks)。"""
    wls = np.array([0.4, 0.6, 1.0])
    ns = np.array([1.45, 1.46, 1.44])
    ks = np.array([0.0, 0.0, 0.0])
    return wls, ns, ks


class TestGetNkAtWavelength:
    def test_vacuum(self):
        nk = get_nk_at_wavelength({}, "Vacuum", 0.532, _load_nk_fake)
        assert nk == 1.0 + 0.0j

    def test_unknown_material(self):
        nk = get_nk_at_wavelength({}, "Unknown", 0.532, _load_nk_fake)
        assert nk == 1.0 + 0.0j

    def test_interpolation(self):
        materials_db = {
            "SiO2": {
                "Shelf ID": "s",
                "Book ID": "b",
                "Page ID": "p",
            }
        }
        nk = get_nk_at_wavelength(
            materials_db, "SiO2", 0.5, _load_nk_fake
        )
        assert abs(np.real(nk) - 1.455) < 0.01
        assert np.imag(nk) == 0.0


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
