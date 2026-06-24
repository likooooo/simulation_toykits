"""Path-based search labels without loading material files."""

from simulation_database.database_ui import leaf_display_label, material_unique_name_from_path


def test_material_unique_name_yml_under_nk() -> None:
    assert material_unique_name_from_path(["main", "Ag", "nk", "Johnson.yml"]) == "Ag(Johnson)"


def test_material_unique_name_air_birch() -> None:
    path = ["other", "mixed gases", "air", "nk", "Birch.yml"]
    assert material_unique_name_from_path(path) == "air(Birch)"


def test_material_unique_name_dir_leaf() -> None:
    assert material_unique_name_from_path(["some", "material_dir"]) == "material_dir"


def test_leaf_display_label_spectrum() -> None:
    assert leaf_display_label("spectra", ["AM1.5G"]) == "AM1.5G"


def test_leaf_display_label_empty_spectrum() -> None:
    assert leaf_display_label("spectra", []) == "spectra"
