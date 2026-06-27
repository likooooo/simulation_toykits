"""common.get_nk_at_wavelength 测试用例"""

from unittest.mock import patch

from common import get_nk_at_wavelength


class _FakeMat:
    def __init__(self, n=1.46, k=0.0):
        self._n = n
        self._k = k

    def nk_at_wavelength_um(self, wl_um):
        return complex(self._n, self._k)


class TestGetNkAtWavelength:
    @patch("common.get_workspace_materials")
    def test_vacuum(self, mock_get_materials):
        mock_get_materials.return_value = {}
        nk = get_nk_at_wavelength("Vacuum", 0.532)
        assert nk == 1.0 + 0.0j

    @patch("common.get_workspace_materials")
    def test_unknown_material(self, mock_get_materials):
        mock_get_materials.return_value = {}
        nk = get_nk_at_wavelength("Unknown", 0.532)
        assert nk == 1.0 + 0.0j

    @patch("common.get_workspace_materials")
    def test_from_database_material(self, mock_get_materials):
        mock_get_materials.return_value = {"SiO2": _FakeMat(1.46, 0.0)}
        nk = get_nk_at_wavelength("SiO2", 0.5)
        assert nk == 1.46 + 0.0j
