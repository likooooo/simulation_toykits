"""Read-only layer table and formula helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from filmstack_simulation.presets import layer_token_from_material
from filmstack_simulation.simulation import nk_at_wavelength


def _material_label(material: Any) -> str:
    return getattr(material, "name", str(material))


def _formula_material_token(material: Any) -> str:
    catalog_fn = getattr(material, "catalog_name", None)
    if callable(catalog_fn):
        return str(catalog_fn())
    return _material_label(material)


def _layer_formula_token(
    material: Any,
    thickness_um: float,
    materials_db: Mapping[str, Any] | None,
    *,
    wl_ref_um: float = 0.55,
) -> str:
    return layer_token_from_material(
        material,
        thickness_um,
        materials_db,
        material_name=_formula_material_token(material),
        wl_ref_um=wl_ref_um,
        nk_at_wavelength=nk_at_wavelength,
    )


def film_layer_indices(n_layers: int) -> list[int]:
    return list(range(1, max(int(n_layers) - 1, 1)))


def layer_bounds_from_ranges(
    film_indices: Sequence[int],
    thicknesses_um: Sequence[float],
    range_pct_by_index: Mapping[int, float],
) -> list[dict[str, float | int]]:
    bounds: list[dict[str, float | int]] = []
    for idx in film_indices:
        t0 = float(thicknesses_um[idx])
        r = float(range_pct_by_index[idx]) / 100.0
        bounds.append(
            {
                "index": int(idx),
                "min": t0 * (1.0 - r),
                "max": t0 * (1.0 + r),
            }
        )
    return bounds


def stack_table_rows(
    materials: Sequence[Any],
    thicknesses_um: Sequence[float],
    *,
    layer_range_pct: Mapping[int, float] | None = None,
    film_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    if film_indices is None:
        film_indices = film_layer_indices(len(materials))
    film_set = set(film_indices)
    rows = []
    for layer_idx, (mat, t) in enumerate(zip(materials, thicknesses_um)):
        label = _material_label(mat)
        if t == 0.0 and (layer_idx == 0 or layer_idx == len(materials) - 1):
            thickness_display = "∞"
        else:
            thickness_display = f"{float(t):.5f}"
        range_pct: float | None = None
        if layer_idx in film_set and layer_range_pct is not None:
            raw = layer_range_pct.get(layer_idx)
            if raw is not None:
                range_pct = float(raw)
        rows.append(
            {
                "_idx": int(layer_idx),
                "材料": label,
                "厚度 (μm)": thickness_display,
                "厚度变化范围 (%)": range_pct,
            }
        )
    return pd.DataFrame(rows)


def sync_layer_range_pct_from_table(
    edited: pd.DataFrame,
    film_indices: Sequence[int],
) -> dict[int, float]:
    out: dict[int, float] = {}
    idx_col = "_idx" if "_idx" in edited.columns else None
    for idx in film_indices:
        if idx_col is not None:
            rows = edited.loc[edited[idx_col] == idx, "厚度变化范围 (%)"]
            if rows.empty:
                continue
            val = rows.iloc[0]
        elif idx in edited.index:
            val = edited.at[idx, "厚度变化范围 (%)"]
        else:
            continue
        if pd.notna(val):
            out[int(idx)] = float(val)
    return out


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
