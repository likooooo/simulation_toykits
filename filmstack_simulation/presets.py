"""Filmstack preset types and formula builders (host supplies PresetCatalog)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Mapping

from filmstack_simulation.simulation import nk_at_wavelength

CUSTOM_PRESET_ID = "custom"


def format_thickness_um(thickness_um: float) -> str:
    t = float(thickness_um)
    if t == 0.0:
        return "0"
    if abs(t) >= 0.01:
        return f"{t:.5f}".rstrip("0").rstrip(".")
    return f"{t:.8f}".rstrip("0").rstrip(".") or "0"


def layer_token_from_parts(
    material: str,
    thickness_um: float,
    *,
    inline_n: float | None = None,
    inline_k: float | None = None,
) -> str:
    token = f"{material} {format_thickness_um(thickness_um)}"
    if inline_n is not None:
        k = inline_k if inline_k is not None else 0.0
        token += f" {inline_n} {k}"
    return token


def layer_token_from_material(
    material: Any,
    thickness_um: float,
    materials_db: Mapping[str, Any] | None,
    *,
    material_name: str,
    wl_ref_um: float,
    nk_at_wavelength=nk_at_wavelength,
) -> str:
    token = f"{material_name} {format_thickness_um(float(thickness_um))}"
    if _needs_inline_nk(material, materials_db, material_name):
        nk = nk_at_wavelength(material, wl_ref_um)
        token += f" {float(nk.real):g} {float(nk.imag):g}"
    return token


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

QwRule = Literal["quarter", "fixed"]

PresetBuilder = Callable[[float, Dict[str, Any]], str]


@dataclass(frozen=True)
class PresetLayer:
    material: str
    thickness_um: float | None = None
    qw_rule: QwRule = "fixed"
    inline_n: float | None = None
    inline_k: float | None = None


@dataclass(frozen=True)
class FilmstackPreset:
    id: str
    label: str
    layers: tuple[PresetLayer, ...] = ()
    period_repeat: tuple[int, int, int] | None = None  # start, end exclusive, count
    custom_builder: PresetBuilder | None = None


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


def _layer_token(layer: PresetLayer, thickness_um: float) -> str:
    return layer_token_from_parts(
        layer.material,
        thickness_um,
        inline_n=layer.inline_n,
        inline_k=layer.inline_k,
    )


def _qw_thickness_um(wl_mid_um: float, n_real: float) -> float:
    if n_real <= 0.0:
        raise ValueError(f"Quarter-wave n={n_real} 无效。")
    return wl_mid_um / (4.0 * n_real)


def _qw_thickness_for_material(
    material: str,
    wl_mid_um: float,
    materials_db: Dict[str, Any],
) -> float:
    mat = materials_db.get(material)
    if mat is None:
        raise ValueError(f"预设材料 '{material}' 不在工作区，请确认 app 已加载默认材料。")
    nk = nk_at_wavelength(mat, wl_mid_um)
    n_real = float(nk.real)
    if n_real <= 0.0:
        raise ValueError(f"材料 '{material}' 在 {wl_mid_um} µm 处 n={n_real} 无效。")
    return _qw_thickness_um(wl_mid_um, n_real)


def _resolve_thickness(
    layer: PresetLayer,
    wl_mid_um: float,
    materials_db: Dict[str, Any],
) -> float:
    if layer.qw_rule == "quarter":
        return _qw_thickness_for_material(layer.material, wl_mid_um, materials_db)
    if layer.thickness_um is None:
        raise ValueError(f"层 '{layer.material}' 缺少固定厚度。")
    return layer.thickness_um


def _build_layer_sequence(
    preset: FilmstackPreset,
    wl_mid_um: float,
    materials_db: Dict[str, Any],
) -> List[str]:
    tokens: List[str] = []
    layers = preset.layers
    i = 0
    while i < len(layers):
        if preset.period_repeat and i == preset.period_repeat[0]:
            start, end, count = preset.period_repeat
            period_tokens: List[str] = []
            for j in range(start, end):
                layer = layers[j]
                t = _resolve_thickness(layer, wl_mid_um, materials_db)
                period_tokens.append(_layer_token(layer, t))
            inner = " ".join(period_tokens)
            tokens.append(f"({inner})^{count}")
            i = end
            continue
        layer = layers[i]
        t = _resolve_thickness(layer, wl_mid_um, materials_db)
        tokens.append(_layer_token(layer, t))
        i += 1
    return tokens


def get_wl_mid_um(
    sim_wl_from: float | None,
    sim_wl_to: float | None,
    fallback_um: float = 0.55,
) -> float:
    if sim_wl_from is not None and sim_wl_to is not None:
        return 0.5 * (float(sim_wl_from) + float(sim_wl_to))
    return fallback_um


def build_formula_for_preset(
    preset_id: str,
    catalog: PresetCatalog,
    materials_db: Dict[str, Any],
    wl_mid_um: float,
) -> str:
    if preset_id == CUSTOM_PRESET_ID:
        raise ValueError("自定义预设无默认公式")
    preset = catalog.preset_by_id.get(preset_id)
    if preset is None:
        raise ValueError(f"未知预设: {preset_id}")
    if preset.custom_builder is not None:
        return preset.custom_builder(wl_mid_um, materials_db)
    return " ".join(_build_layer_sequence(preset, wl_mid_um, materials_db))
