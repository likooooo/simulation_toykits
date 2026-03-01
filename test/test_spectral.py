"""core.spectral 测试用例（不依赖 assets.simulation 的部分）"""

import numpy as np
import pytest
from core.spectral import build_nk_map_for_wavelengths


class TestBuildNkMapForWavelengths:
    def test_all_in_db_uses_get_nk_fn(self):
        wls = np.array([0.5, 0.6])
        materials_db = {"A": {}, "B": {}}
        def get_nk(name, w):
            return (1.5 + 0.1j) if name == "A" else (2.0 + 0.0j)
        nk_map, not_in_db = build_nk_map_for_wavelengths(
            ["A", "B", "A"],
            [1.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            wls,
            materials_db,
            get_nk,
        )
        assert len(not_in_db) == 0
        assert "A" in nk_map and "B" in nk_map
        assert len(nk_map["A"]) == 2
        assert len(nk_map["B"]) == 2
        assert nk_map["A"][0] == 1.5 + 0.1j
        assert nk_map["B"][1] == 2.0 + 0.0j

    def test_not_in_db_uses_constant_nk(self):
        wls = np.linspace(0.4, 0.8, 3)
        materials_db = {}
        nk_map, not_in_db = build_nk_map_for_wavelengths(
            ["Custom"],
            [1.7],
            [0.01],
            wls,
            materials_db,
            lambda n, w: 1.0 + 0.0j,
        )
        assert "Custom" in not_in_db
        assert len(nk_map["Custom"]) == 3
        assert all(nk == 1.7 + 0.01j for nk in nk_map["Custom"])

    def test_mixed_in_and_not_in_db(self):
        wls = np.array([0.55])
        materials_db = {"SiO2": {}}
        def get_nk(name, w):
            return 1.46 + 0.0j
        nk_map, not_in_db = build_nk_map_for_wavelengths(
            ["SiO2", "Custom", "SiO2"],
            [1.46, 2.1, 1.46],
            [0.0, 0.0, 0.0],
            wls,
            materials_db,
            get_nk,
        )
        assert "Custom" in not_in_db
        assert "SiO2" not in not_in_db
        assert nk_map["SiO2"][0] == 1.46 + 0.0j
        assert nk_map["Custom"][0] == 2.1 + 0.0j
