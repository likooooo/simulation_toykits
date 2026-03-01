"""core.material_index 测试用例"""

import tempfile
import os
import pytest
import pandas as pd
from core.material_index import (
    clean_name,
    load_material_index,
    load_material_index_cached,
    _cache_path_for_csv,
)


class TestCleanName:
    def test_identity(self):
        assert clean_name("SiO2") == "SiO2"

    def test_sub(self):
        assert clean_name("H<sub>2</sub>O") == "H2O"

    def test_sup_removed(self):
        assert clean_name("x<sup>2</sup>") == "x"

    def test_non_string(self):
        assert clean_name(None) is None


class TestLoadMaterialIndex:
    def test_load_csv(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            f.write("shelf_id,shelf_name,book_id,book_name,page_id,page_name\n")
            f.write('1,Main,10,SiO2,100,Malitson\n')
            f.write('1,Main,11,H<sub>2</sub>O,101,Page1\n')
            path = f.name
        try:
            df = load_material_index(path)
            assert len(df) == 2
            assert df["book_name"].iloc[1] == "H2O"
        finally:
            os.unlink(path)


class TestLoadMaterialIndexCached:
    def test_cache_path(self):
        assert _cache_path_for_csv("/a/b/materials_index.csv").endswith(
            "materials_index.cache.pkl"
        )
        assert _cache_path_for_csv("m.csv") == "m.cache.pkl"

    def test_first_load_creates_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "idx.csv")
            with open(csv_path, "w", newline="") as f:
                f.write("shelf_id,shelf_name,book_id,book_name,page_id,page_name\n")
                f.write("1,Shelf,10,BookA,100,PageA\n")
            cache_path = _cache_path_for_csv(csv_path)
            assert not os.path.isfile(cache_path)
            df1 = load_material_index_cached(csv_path)
            assert os.path.isfile(cache_path)
            assert len(df1) == 1
            assert "book_name" in df1.columns

    def test_second_load_uses_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "idx.csv")
            with open(csv_path, "w", newline="") as f:
                f.write("shelf_id,shelf_name,book_id,book_name,page_id,page_name\n")
                f.write("1,S,10,B1,100,P1\n")
                f.write("1,S,11,B2,101,P2\n")
            df1 = load_material_index_cached(csv_path)
            df2 = load_material_index_cached(csv_path)
            pd.testing.assert_frame_equal(df1, df2)
            assert list(df2["book_name"]) == ["B1", "B2"]

    def test_no_cache_reparses_after_csv_change(self):
        """Cache不存在或已删时，会重新解析 CSV。"""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "idx.csv")
            cache_path = _cache_path_for_csv(csv_path)
            with open(csv_path, "w", newline="") as f:
                f.write("shelf_id,shelf_name,book_id,book_name,page_id,page_name\n")
                f.write("1,S,10,Old,100,P\n")
            load_material_index_cached(csv_path)
            os.remove(cache_path)
            with open(csv_path, "w", newline="") as f:
                f.write("shelf_id,shelf_name,book_id,book_name,page_id,page_name\n")
                f.write("1,S,10,New,100,P\n")
            df = load_material_index_cached(csv_path)
            assert df["book_name"].iloc[0] == "New"
