"""Read-only layer table and formula helpers."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import pandas as pd

import filmstack_visualizer
from filmstack_simulation.presets import format_thickness_um, layer_token_from_material
from filmstack_simulation.simulation import nk_at_wavelength


def _material_label(material: Any) -> str:
    return getattr(material, "name", str(material))


def _formula_material_token(material: Any) -> str:
    unique_fn = getattr(material, "unique_name", None)
    if callable(unique_fn):
        return str(unique_fn())
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


def _expand_formula_repeats(formula: str) -> str:
    while "(" in formula:
        match = re.search(r"\(([^()]+)\)\^(\d+)", formula)
        if not match:
            break
        content, times = match.group(1), int(match.group(2))
        formula = formula.replace(match.group(0), (content + " ") * times)
    return formula


def _formula_layer_thickness_token_indices(tokens: Sequence[str]) -> list[tuple[int, bool]]:
    """For each parsed layer: (thickness_token_index, incoherent)."""
    infos: list[tuple[int, bool]] = []
    i = 0
    while i < len(tokens):
        if i + 1 >= len(tokens):
            break
        token = tokens[i]
        if token.startswith("["):
            _, incoherent = filmstack_visualizer._parse_thickness_um(tokens[i + 1])
            infos.append((i + 1, incoherent))
            i += 2
            continue
        _, incoherent = filmstack_visualizer._parse_thickness_um(tokens[i + 1])
        infos.append((i + 1, incoherent))
        if i + 3 < len(tokens):
            try:
                float(tokens[i + 2])
                float(tokens[i + 3])
                i += 4
                continue
            except ValueError:
                pass
        i += 2
    return infos


def apply_optimized_thicknesses_to_formula(
    formula: str,
    film_indices: Sequence[int],
    thicknesses_um: Sequence[float],
) -> str:
    """Patch only optimizable layer thickness tokens; preserve MG brackets, *, inline n/k."""
    expanded = _expand_formula_repeats(formula.strip())
    parsed = filmstack_visualizer.parse_filmstack_formula_v1(expanded)
    thicknesses_parsed = [float(layer["Thickness (um)"]) for layer in parsed]
    leading_bookend = thicknesses_parsed[0] != 0.0

    tokens = list(filmstack_visualizer._tokenize_filmstack_formula(expanded))
    layer_tokens = _formula_layer_thickness_token_indices(tokens)
    if len(layer_tokens) != len(parsed):
        raise ValueError(
            f"formula layer token count {len(layer_tokens)} != parsed layers {len(parsed)}"
        )

    for exp_idx in film_indices:
        parsed_idx = int(exp_idx) - (1 if leading_bookend else 0)
        if parsed_idx < 0 or parsed_idx >= len(parsed):
            raise ValueError(
                f"film index {exp_idx} maps to invalid parsed layer {parsed_idx}"
            )
        tok_idx, incoherent = layer_tokens[parsed_idx]
        new_t = float(thicknesses_um[exp_idx])
        suffix = "*" if incoherent else ""
        tokens[tok_idx] = format_thickness_um(new_t) + suffix

    return " ".join(tokens)


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
