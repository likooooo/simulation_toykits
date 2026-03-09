"""core.formula 测试用例"""

import pytest
from core.formula import parse_formula, parse_formula_v1


class TestParseFormula:
    def test_simple_pairs(self):
        got = parse_formula("SiO2 0.1 TiO2 0.2")
        assert len(got) == 2
        assert got[0] == {"Material": "SiO2", "Thickness (um)": 0.1}
        assert got[1] == {"Material": "TiO2", "Thickness (um)": 0.2}

    def test_periodic(self):
        got = parse_formula("(H 0.1 L 0.2)^2")
        assert len(got) == 4
        assert got[0]["Material"] == "H"
        assert got[0]["Thickness (um)"] == 0.1
        assert got[2]["Material"] == "H"


class TestParseFormulaV1:
    def test_material_thickness_only(self):
        got = parse_formula_v1("Vacuum 0 SiO2 0.5")
        assert len(got) == 2
        assert got[0] == {
            "Material": "Vacuum",
            "Thickness (um)": 0.0,
            "n": None,
            "k": None,
        }
        assert got[1] == {
            "Material": "SiO2",
            "Thickness (um)": 0.5,
            "n": None,
            "k": None,
        }

    def test_material_thickness_nk(self):
        got = parse_formula_v1("SiO2 0.1 1.46 0.001")
        assert len(got) == 1
        assert got[0]["Material"] == "SiO2"
        assert got[0]["Thickness (um)"] == 0.1
        assert got[0]["n"] == 1.46
        assert got[0]["k"] == 0.001

    def test_periodic_v1(self):
        got = parse_formula_v1("(SiO2 0.1 Ta2O5 0.02)^2")
        assert len(got) == 4
        assert got[0]["Material"] == "SiO2"
        assert got[1]["Material"] == "Ta2O5"
        assert got[2]["Material"] == "SiO2"
        assert got[3]["Material"] == "Ta2O5"

    def test_empty_tokens_ignored(self):
        got = parse_formula_v1("  Vacuum 0   SiO2 0.1  ")
        assert len(got) == 2
        assert got[0]["Material"] == "Vacuum"
        assert got[1]["Material"] == "SiO2"

    def test_multiple_layers_with_and_without_nk(self):
        # A 0.1 1.5 0.0  B 0.2  (no n,k)  C 0.05 2.0 0.001
        got = parse_formula_v1("A 0.1 1.5 0.0 B 0.2 C 0.05 2.0 0.001")
        assert len(got) == 3
        assert got[0]["Material"] == "A" and got[0]["n"] == 1.5 and got[0]["k"] == 0.0
        assert got[1]["Material"] == "B" and got[1]["n"] is None and got[1]["k"] is None
        assert got[2]["Material"] == "C" and got[2]["n"] == 2.0 and got[2]["k"] == 0.001

    def test_periodic_large_repeat(self):
        got = parse_formula_v1("(H 0.01 L 0.02)^5")
        assert len(got) == 10
        assert all(got[i]["Material"] == "H" for i in (0, 2, 4, 6, 8))
        assert all(got[i]["Material"] == "L" for i in (1, 3, 5, 7, 9))
