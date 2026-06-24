"""Read-only layer table and formula helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from filmstack_simulation.simulation import nk_at_wavelength


def _material_label(material: Any) -> str:
    return getattr(material, "name", str(material))


def _formula_material_token(material: Any) -> str:
    catalog_fn = getattr(material, "catalog_name", None)
    if callable(catalog_fn):
        return str(catalog_fn())
    return _material_label(material)


def _needs_inline_nk(material: Any, materials_db: Mapping[str, Any] | None) -> bool:
    if not materials_db:
        return True
    name = _formula_material_token(material)
    db_mat = materials_db.get(name)
    if db_mat is not None and db_mat is material:
        return False
    for obj in materials_db.values():
        if obj is material:
            return False
    return True


def _format_thickness_um(thickness_um: float) -> str:
    t = float(thickness_um)
    if t == 0.0:
        return "0"
    if abs(t) >= 0.01:
        return f"{t:.5f}".rstrip("0").rstrip(".")
    return f"{t:.8f}".rstrip("0").rstrip(".") or "0"


def _layer_formula_token(
    material: Any,
    thickness_um: float,
    materials_db: Mapping[str, Any] | None,
    *,
    wl_ref_um: float = 0.55,
) -> str:
    token = f"{_formula_material_token(material)} {_format_thickness_um(float(thickness_um))}"
    if _needs_inline_nk(material, materials_db):
        nk = nk_at_wavelength(material, wl_ref_um)
        token += f" {float(nk.real):g} {float(nk.imag):g}"
    return token


def stack_table_rows(
    materials: Sequence[Any],
    thicknesses_um: Sequence[float],
) -> pd.DataFrame:
    rows = []
    for mat, t in zip(materials, thicknesses_um):
        label = _material_label(mat)
        if t == 0.0 and (len(rows) == 0 or len(rows) == len(materials) - 1):
            thickness_display = "∞"
        else:
            thickness_display = f"{float(t):.5f}"
        rows.append({"材料": label, "厚度 (μm)": thickness_display})
    return pd.DataFrame(rows)


def formula_from_stack(
    materials: Sequence[Any],
    thicknesses_um: Sequence[float],
    materials_db: Mapping[str, Any] | None = None,
    *,
    wl_ref_um: float = 0.55,
) -> str:
    parts = [
        _layer_formula_token(mat, t, materials_db, wl_ref_um=wl_ref_um)
        for mat, t in zip(materials, thicknesses_um)
    ]
    return " ".join(parts)
