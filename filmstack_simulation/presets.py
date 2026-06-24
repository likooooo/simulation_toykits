"""Predefined filmstack formulas aligned with TMM / Oghma sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal

from filmstack_simulation.simulation import nk_at_wavelength

CUSTOM_PRESET_ID = "custom"

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


def _fmt_thickness(t_um: float) -> str:
    return f"{t_um:.5f}"


def _layer_token(layer: PresetLayer, thickness_um: float) -> str:
    if layer.inline_n is not None:
        k = layer.inline_k if layer.inline_k is not None else 0.0
        return f"{layer.material} {_fmt_thickness(thickness_um)} {layer.inline_n} {k}"
    return f"{layer.material} {_fmt_thickness(thickness_um)}"


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


_BRAGG_PERIOD_UM = 0.209
_BRAGG_H_N = 1.5


def _build_bragg_formula(wl_mid_um: float, materials_db: Dict[str, Any]) -> str:
    del materials_db
    t_h = _qw_thickness_um(wl_mid_um, _BRAGG_H_N)
    t_l = _BRAGG_PERIOD_UM - t_h
    if t_l <= 0.0:
        raise ValueError("Bragg 周期内低折射率层厚度 ≤ 0，请调整波长或材料。")
    inner = f"H {_fmt_thickness(t_h)} {_BRAGG_H_N} 0.0 air {_fmt_thickness(t_l)}"
    return f"air 0.00000 ({inner})^30 Si 0.00000"


# --- preset definitions (audit §2) ---

_AR_QW_SI = FilmstackPreset(
    id="ar_qw_si",
    label="减反膜系",
    period_repeat=(1, 3, 2),
    layers=(
        PresetLayer("air", 0.0),
        PresetLayer("SiO2", qw_rule="quarter"),
        PresetLayer("Ta2O5", qw_rule="quarter"),
        PresetLayer("Si", 0.0),
    ),
)

_BRAGG_MIRROR = FilmstackPreset(
    id="bragg_mirror",
    label="布拉格反射镜",
    custom_builder=_build_bragg_formula,
)

_OPTICAL_FILTER = FilmstackPreset(
    id="optical_filter",
    label="光学滤光片",
    period_repeat=(2, 4, 5),
    layers=(
        PresetLayer("air", 0.0),
        PresetLayer("air", 0.1),
        PresetLayer("H", 0.083, inline_n=1.5, inline_k=0.0),
        PresetLayer("L", 0.126, inline_n=1.5, inline_k=0.0),
        PresetLayer("Exit", 0.126, inline_n=1.0, inline_k=0.0),
        PresetLayer("Si", 0.0),
    ),
)

_FABRY_PEROT = FilmstackPreset(
    id="fabry_perot",
    label="FP 共振腔",
    layers=(
        PresetLayer("air", 0.0),
        PresetLayer("Mirror", 0.01, inline_n=3.0, inline_k=0.0),
        PresetLayer("air", 0.25),
        PresetLayer("Mirror", 0.01, inline_n=3.0, inline_k=0.0),
        PresetLayer("Si", 0.0),
    ),
)

_OLED_ITO_AL = FilmstackPreset(
    id="oled_ito_al",
    label="OLED 栈",
    layers=(
        PresetLayer("ito", 0.0),
        PresetLayer("ito", 0.15),
        PresetLayer("NPD", 0.04),
        PresetLayer("Alq3", 0.03),
        PresetLayer("TPBi", 0.03),
        PresetLayer("LiF", 0.01),
        PresetLayer("std", 0.05),
        PresetLayer("std", 0.0),
    ),
)

_SPR_BK7_CR_AU = FilmstackPreset(
    id="spr_bk7_cr_au",
    label="SPR 金属膜",
    layers=(
        PresetLayer("BK7", 0.0, inline_n=1.517, inline_k=0.0),
        PresetLayer("Cr", 0.005, inline_n=3.719, inline_k=4.362),
        PresetLayer("Au", 0.03, inline_n=0.130, inline_k=3.162),
        PresetLayer("air", 0.0, inline_n=1.0, inline_k=0.0),
    ),
)

_PAPER_SIO2_SI = FilmstackPreset(
    id="paper_sio2_si",
    label="椭偏基准",
    layers=(
        PresetLayer("air", 0.0, inline_n=1.0, inline_k=0.0),
        PresetLayer("SiO2", 0.05, inline_n=1.46, inline_k=0.0),
        PresetLayer("Si", 0.0, inline_n=3.87, inline_k=0.02),
    ),
)

PRESETS: tuple[FilmstackPreset, ...] = (
    _AR_QW_SI,
    _BRAGG_MIRROR,
    _OPTICAL_FILTER,
    _FABRY_PEROT,
    _OLED_ITO_AL,
    _SPR_BK7_CR_AU,
    _PAPER_SIO2_SI,
)

PRESET_BY_ID: Dict[str, FilmstackPreset] = {p.id: p for p in PRESETS}

PRESET_SELECT_OPTIONS: list[tuple[str, str]] = [
    (CUSTOM_PRESET_ID, "自定义"),
] + [(p.id, p.label) for p in PRESETS]

PRESET_IDS: list[str] = [pid for pid, _ in PRESET_SELECT_OPTIONS]
VALID_PRESET_IDS: frozenset[str] = frozenset(PRESET_IDS)
PRESET_LABELS: dict[str, str] = dict(PRESET_SELECT_OPTIONS)

DEFAULT_PRESET_ID = "ar_qw_si"


def build_formula_for_preset(
    preset_id: str,
    materials_db: Dict[str, Any],
    wl_mid_um: float,
) -> str:
    if preset_id == CUSTOM_PRESET_ID:
        raise ValueError("自定义预设无默认公式")
    preset = PRESET_BY_ID.get(preset_id)
    if preset is None:
        raise ValueError(f"未知预设: {preset_id}")
    if preset.custom_builder is not None:
        return preset.custom_builder(wl_mid_um, materials_db)
    return " ".join(_build_layer_sequence(preset, wl_mid_um, materials_db))
