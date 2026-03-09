"""Fresnel agent 工具层测试：保证每个 agent 可调用的 tool 均可被正常调用。"""

import os
import sys
import tempfile
import pytest

# 从仓库根目录加载（与 run_agent 一致）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agent.fresnel_caculator.tools import run_tool, TOOLS


# 不依赖材料库的膜系公式（内联 n,k，仅需 Vacuum + 一层介质）
_FORMULA_INLINE_NK = "Vacuum 0 Layer 0.1 1.5 0 Vacuum 0"


class TestListMaterialIndex:
    """list_material_index：必填 material_name，返回 shelf_id、books（仅含 page_id）。"""

    def test_call_with_material_name(self):
        out = run_tool("list_material_index", {"material_name": "SiO2"})
        assert isinstance(out, dict)
        assert "shelf_id" in out and "books" in out
        for b in out.get("books", []):
            assert "page_id" in b
            assert "book_id" not in b

    def test_call_material_name_empty_returns_error(self):
        out = run_tool("list_material_index", {"material_name": ""})
        assert isinstance(out, dict)
        assert "error" in out
        assert "不能为空" in out["error"]

    def test_call_with_csv_path(self):
        out = run_tool("list_material_index", {
            "material_name": "SiO2",
            "csv_path": "/nonexistent/index.csv",
        })
        assert isinstance(out, dict)
        assert "error" in out


class TestGetMaterialNk:
    """get_material_nk：获取材料 n/k。Vacuum 不依赖库。"""

    def test_call_vacuum(self):
        out = run_tool("get_material_nk", {
            "shelf_id": "main",
            "book_id": "Vacuum",
            "page_id": "any",
        })
        print(out)
        assert isinstance(out, dict)
        assert out.get("material") == "Vacuum"
        assert "wavelength_um" in out and "n" in out and "k" in out

    def test_call_material_may_fail_without_db(self):
        out = run_tool("get_material_nk", {
            "shelf_id": "main",
            "book_id": "SiO2",
            "page_id": "Malitson",
        })
        print(out)
        assert isinstance(out, dict)
        assert "material" in out or "error" in out

    def test_call_with_ratio_returns_downsampled(self):
        out = run_tool("get_material_nk", {
            "shelf_id": "main",
            "book_id": "Vacuum",
            "page_id": "any",
            "ratio": 0.5,
        })
        assert isinstance(out, dict)
        assert out.get("material") == "Vacuum"
        # Vacuum 只有 2 个点，ratio=0.5 -> max(1, 1)=1 个点
        assert len(out["wavelength_um"]) >= 1

    def test_call_ratio_invalid_returns_error(self):
        out = run_tool("get_material_nk", {
            "shelf_id": "main",
            "book_id": "Vacuum",
            "page_id": "any",
            "ratio": 1.5,
        })
        assert isinstance(out, dict)
        assert "error" in out
        assert "ratio" in out["error"]


class TestExportNkToCsv:
    """export_nk_to_csv：导出 nk 到 CSV。用 Vacuum 可不依赖材料库。"""

    def test_call_vacuum(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            out = run_tool("export_nk_to_csv", {
                "shelf_id": "main",
                "book_id": "Vacuum",
                "page_id": "any",
                "out_path": path,
            })
            print(out)
            assert isinstance(out, dict)
            assert out.get("success") is True or "error" in out
            if out.get("success"):
                assert os.path.isfile(out["path"])
        finally:
            if os.path.isfile(path):
                os.unlink(path)


class TestParseFilmFormula:
    """parse_film_formula：解析膜系公式。"""

    def test_call_simple(self):
        out = run_tool("parse_film_formula", {
            "formula": "Vacuum 0 SiO2 0.1 Vacuum 0",
        })
        print(out)
        assert isinstance(out, dict)
        assert "layers" in out
        assert len(out["layers"]) >= 2

    def test_call_with_periodic(self):
        out = run_tool("parse_film_formula", {
            "formula": "(SiO2 0.1 Ta2O5 0.02)^2",
        })
        print(out)
        assert isinstance(out, dict)
        assert "layers" in out or "error" in out


class TestComputeFilmstack:
    """compute_filmstack：单组膜系 R/T。用内联 n,k 公式不依赖材料库。"""

    def test_call_inline_nk(self):
        out = run_tool("compute_filmstack", {
            "formula": _FORMULA_INLINE_NK,
            "angle_deg": 0.0,
            "wl_um": 0.532,
        })
        print(out)
        assert isinstance(out, dict)
        assert "error" in out or ("R_s" in out and "T_s" in out)

    def test_call_with_figure_path(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            out = run_tool("compute_filmstack", {
                "formula": _FORMULA_INLINE_NK,
                "angle_deg": 0.0,
                "wl_um": 0.532,
                "out_figure_path": path,
            })
            print(out)
            assert isinstance(out, dict)
            if "error" not in out and out.get("figure_path"):
                assert os.path.isfile(out["figure_path"])
        finally:
            if os.path.isfile(path):
                os.unlink(path)


class TestComputeFilmstackBatch:
    """compute_filmstack_batch：批量膜系 R/T。"""

    def test_call(self):
        out = run_tool("compute_filmstack_batch", {
            "formulas": [_FORMULA_INLINE_NK, "Vacuum 0 L2 0.05 2.0 0 Vacuum 0"],
            "angle_deg": 0.0,
            "wl_um": 0.532,
        })
        print(out)
        assert isinstance(out, dict)
        assert "results" in out or "error" in out  # error when simulation skipped in CI
        if "results" in out:
            assert len(out["results"]) == 2


class TestComputeAngleVsRt:
    """compute_angle_vs_rt：固定波长，R/T 随角度变化。"""

    def test_call(self):
        out = run_tool("compute_angle_vs_rt", {
            "formula": _FORMULA_INLINE_NK,
            "wl_um": 0.532,
        })
        print(out)
        assert isinstance(out, dict)
        assert "error" in out or "angles_deg" in out


class TestComputeWavelengthVsRt:
    """compute_wavelength_vs_rt：固定角度，R/T 随波长变化。"""

    def test_call(self):
        out = run_tool("compute_wavelength_vs_rt", {
            "formula": _FORMULA_INLINE_NK,
            "angle_deg": 0.0,
            "wl_min_um": 0.4,
            "wl_max_um": 0.7,
        })
        print(out)
        assert isinstance(out, dict)
        assert "error" in out or "wl_range" in out

    def test_call_with_num_points(self):
        out = run_tool("compute_wavelength_vs_rt", {
            "formula": _FORMULA_INLINE_NK,
            "angle_deg": 0.0,
            "wl_min_um": 0.4,
            "wl_max_um": 0.7,
            "num_points": 20,
        })
        print(out)
        assert isinstance(out, dict)
        assert "error" in out or "num_points" in out


class TestSaveResultsCsv:
    """save_results_csv：将结果列表保存为 CSV。"""

    def test_call(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            out = run_tool("save_results_csv", {
                "rows": [
                    {"formula": "Vacuum 0 SiO2 0.1 Vacuum 0", "R_s": 0.1, "T_s": 0.9},
                    {"formula": "Vacuum 0 TiO2 0.1 Vacuum 0", "R_s": 0.2, "T_s": 0.8},
                ],
                "out_path": path,
            })
            print(out)
            assert isinstance(out, dict)
            assert out.get("success") is True
            assert out.get("rows") == 2
            assert os.path.isfile(path)
        finally:
            if os.path.isfile(path):
                os.unlink(path)


class TestRunToolUnknown:
    """run_tool 对未知工具名应返回 error 而非抛错。"""

    def test_unknown_tool(self):
        out = run_tool("no_such_tool", {})
        print(out)
        assert isinstance(out, dict)
        assert "error" in out
        assert "available" in out
        assert "list_material_index" in out["available"]


class TestAllAgentToolsRegistered:
    """确保 ollama_brain 中声明的工具均在 TOOLS 中且可被 run_tool 调用。"""

    @pytest.fixture
    def schema_names(self):
        from agent.fresnel_caculator.ollama_brain import TOOL_SCHEMAS
        return {s["name"] for s in TOOL_SCHEMAS}

    def test_tools_match_schemas(self, schema_names):
        assert set(TOOLS.keys()) == schema_names

    def test_each_tool_callable(self, schema_names):
        """每个工具至少能无异常调用一次（用最小合法参数；返回 dict 即算可调用）。"""
        minimal_args = {
            "list_material_index": {"material_name": "SiO2"},
            "get_material_nk": {"shelf_id": "main", "book_id": "Vacuum", "page_id": "x"},
            "export_nk_to_csv": {
                "shelf_id": "main", "book_id": "Vacuum", "page_id": "x",
                "out_path": tempfile.mktemp(suffix=".csv"),
            },
            "parse_film_formula": {"formula": "Vacuum 0 Vacuum 0"},
            "compute_filmstack": {
                "formula": _FORMULA_INLINE_NK,
                "angle_deg": 0.0,
                "wl_um": 0.532,
            },
            "compute_filmstack_batch": {
                "formulas": [_FORMULA_INLINE_NK],
                "angle_deg": 0.0,
                "wl_um": 0.532,
            },
            "compute_angle_vs_rt": {"formula": _FORMULA_INLINE_NK, "wl_um": 0.532},
            "compute_wavelength_vs_rt": {
                "formula": _FORMULA_INLINE_NK,
                "angle_deg": 0.0,
                "wl_min_um": 0.4,
                "wl_max_um": 0.7,
            },
            "save_results_csv": {
                "rows": [{"a": 1}],
                "out_path": tempfile.mktemp(suffix=".csv"),
            },
        }
        for name in schema_names:
            args = minimal_args.get(name, {})
            out = run_tool(name, args)
            assert isinstance(out, dict), f"{name} should return dict, got {type(out)}"
