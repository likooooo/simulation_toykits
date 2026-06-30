"""Filmstack preset types (host supplies PresetCatalog from filmstack_templates.json)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

CUSTOM_PRESET_ID = "custom"


def format_thickness_um(thickness_um: float) -> str:
    t = float(thickness_um)
    if t == 0.0:
        return "0"
    if abs(t) >= 0.01:
        return f"{t:.5f}".rstrip("0").rstrip(".")
    return f"{t:.8f}".rstrip("0").rstrip(".") or "0"


def _needs_inline_nk(material: Any, materials_db: Mapping[str, Any] | None, material_name: str) -> bool:
    if not materials_db:
        return True
    db_mat = materials_db.get(material_name)
    if db_mat is not None and db_mat is material:
        return False
    for obj in materials_db.values():
        if obj is material:
            return False
    return True


def layer_token_from_material(
    material: Any,
    thickness_um: float,
    materials_db: Mapping[str, Any] | None,
    *,
    material_name: str,
    wl_ref_um: float,
    nk_at_wavelength=None,
) -> str:
    if nk_at_wavelength is None:
        from filmstack_simulation.simulation import nk_at_wavelength as _nk_at_wavelength

        nk_at_wavelength = _nk_at_wavelength
    token = f"{material_name} {format_thickness_um(float(thickness_um))}"
    if _needs_inline_nk(material, materials_db, material_name):
        nk = nk_at_wavelength(material, wl_ref_um)
        token += f" {float(nk.real):g} {float(nk.imag):g}"
    return token


@dataclass(frozen=True)
class FilmstackPreset:
    id: str
    label: str
    formula: str = ""


@dataclass(frozen=True)
class PresetCatalog:
    presets: tuple[FilmstackPreset, ...]
    default_preset_id: str

    @property
    def preset_by_id(self) -> Dict[str, FilmstackPreset]:
        return {p.id: p for p in self.presets}

    @property
    def preset_select_options(self) -> list[tuple[str, str]]:
        return [(CUSTOM_PRESET_ID, "自定义")] + [(p.id, p.label) for p in self.presets]

    @property
    def preset_ids(self) -> tuple[str, ...]:
        return tuple(pid for pid, _ in self.preset_select_options)

    @property
    def preset_labels(self) -> dict[str, str]:
        return dict(self.preset_select_options)

    @property
    def valid_preset_ids(self) -> frozenset[str]:
        return frozenset(self.preset_ids)


def build_formula_for_preset(
    preset_id: str,
    catalog: PresetCatalog,
) -> str:
    if preset_id == CUSTOM_PRESET_ID:
        raise ValueError("自定义预设无默认公式")
    preset = catalog.preset_by_id.get(preset_id)
    if preset is None:
        raise ValueError(f"未知预设: {preset_id}")
    return preset.formula
