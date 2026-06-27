"""Host-level defaults for simulation_toykits (workspace paths, presets, wavelength range)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import simulation  # noqa: F401 — must load before simulation_database_parser
import simulation_database_parser as sdp

from filmstack_simulation.filmstack_optimization.local_search.opt_config import (
    get_freehand_initial_formula,
    get_freehand_initial_preset_id,
)
from filmstack_simulation.presets import FilmstackPreset, PresetCatalog, PresetLayer

DEFAULT_SPECTRUM_PATH = sdp.spectrum_query_keys("AM1.5G")

# Full query paths from simulation_database root.
_RII_MATERIALS = sdp.RII_QUERY_PREFIX
DEFAULT_MATERIAL_PATH_KEYS: list[list[str]] = [
    # bookend air — Ciddor 1996 standard air (catalog ``air``); 0.23–1.69 µm
    [
        *_RII_MATERIALS,
        "other",
        "mixed gases",
        "air",
        "nk",
        "Ciddor.yml",
    ],
    # ar_qw_si — sdp.DEFAULT_RII_FILMSTACK_MATERIAL_PATHS
    *list(sdp.DEFAULT_RII_FILMSTACK_MATERIAL_PATHS.values()),
    # freehand initial stack — MgF2/TiO2 on N-BK7
    [*_RII_MATERIALS, "specs", "schott", "optical", "N-BK7.yml"],
    [*_RII_MATERIALS, "main", "MgF2", "nk", "Dodge-o.yml"],
    [*_RII_MATERIALS, "main", "TiO2", "nk", "Jolivet-anatase.yml"],
    # bragg / optical_filter / fabry_perot use inline nk (H/L/Exit/Mirror); no generic/n/* in workspace
    # oled_ito_al — 02_oled_tmm Oghma paths
    ["oghma_database", "materials", "oxides", "ITO", "ito.yml"],
    ["oghma_database", "materials", "small_molecules", "NPD.yml"],
    ["oghma_database", "materials", "small_molecules", "Alq3.yml"],
    ["oghma_database", "materials", "small_molecules", "TPBi.yml"],
    [*_RII_MATERIALS, "main", "LiF", "nk", "Li.yml"],
    ["oghma_database", "materials", "metal", "Al", "std.yml"],
]

RECOMMENDED_SIM_WL_FROM_UM = 0.38
RECOMMENDED_SIM_WL_TO_UM = 0.78

BRAGG_PERIOD_UM = 0.209
BRAGG_H_N = 1.5
BRAGG_PERIOD_REPEAT = 8


def _fmt_thickness(t_um: float) -> str:
    return f"{t_um:.5f}"


def _qw_thickness_um(wl_mid_um: float, n_real: float) -> float:
    if n_real <= 0.0:
        raise ValueError(f"Quarter-wave n={n_real} 无效。")
    return wl_mid_um / (4.0 * n_real)


def build_bragg_formula(wl_mid_um: float, materials_db: Dict[str, Any]) -> str:
    del materials_db
    t_h = _qw_thickness_um(wl_mid_um, BRAGG_H_N)
    t_l = BRAGG_PERIOD_UM - t_h
    if t_l <= 0.0:
        raise ValueError("Bragg 周期内低折射率层厚度 ≤ 0，请调整波长或材料。")
    inner = f"H {_fmt_thickness(t_h)} {BRAGG_H_N} 0.0 air {_fmt_thickness(t_l)}"
    return f"air 0.00000 ({inner})^{BRAGG_PERIOD_REPEAT} Si 0.00000"


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
    custom_builder=build_bragg_formula,
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

DEFAULT_PRESET_ID = "ar_qw_si"

FILMSTACK_PRESET_CATALOG = PresetCatalog(
    presets=PRESETS,
    default_preset_id=DEFAULT_PRESET_ID,
)


@dataclass(frozen=True)
class FilmstackInitialDefaults:
    preset_id: str
    formula: str
    wl_from_um: float
    wl_to_um: float


def resolve_filmstack_initial_defaults(
    valid_preset_ids: frozenset[str],
) -> FilmstackInitialDefaults:
    return FilmstackInitialDefaults(
        preset_id=get_freehand_initial_preset_id(valid_preset_ids),
        formula=get_freehand_initial_formula(),
        wl_from_um=RECOMMENDED_SIM_WL_FROM_UM,
        wl_to_um=RECOMMENDED_SIM_WL_TO_UM,
    )
