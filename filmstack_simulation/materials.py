"""Filmstack default material path keys and simulation wavelength constants."""

from __future__ import annotations

# Path keys relative to simulation_core/assets/database/materials/
DEFAULT_MATERIAL_PATH_KEYS: list[list[str]] = [
    # bookend air — Ciddor 1996 standard air (catalog ``air``); 0.23–1.69 µm
    [
        "refractive_index_info",
        "other",
        "mixed gases",
        "air",
        "nk",
        "Ciddor.yml",
    ],
    # ar_qw_si — filmstack_visualizer DEFAULT_FILMSTACK_MATERIALS_DB
    ["refractive_index_info", "main", "SiO2", "nk", "Arosa.yml"],
    ["refractive_index_info", "main", "Ta2O5", "nk", "Cheikh-amorphous-3.28-8-450.yml"],
    ["refractive_index_info", "main", "Si", "nk", "Aspnes.yml"],
    # freehand initial stack — MgF2/TiO2 on N-BK7
    ["refractive_index_info", "specs", "schott", "optical", "N-BK7.yml"],
    ["refractive_index_info", "main", "MgF2", "nk", "Dodge-o.yml"],
    ["refractive_index_info", "main", "TiO2", "nk", "Jolivet-anatase.yml"],
    # bragg / optical_filter / fabry_perot use inline nk (H/L/Exit/Mirror); no generic/n/* in workspace
    # oled_ito_al — 02_oled_tmm Oghma paths
    ["oxides", "ITO", "ito"],
    ["small_molecules", "NPD"],
    ["small_molecules", "Alq3"],
    ["small_molecules", "TPBi"],
    ["refractive_index_info", "main", "LiF", "nk", "Li.yml"],
    ["metal", "Al", "std"],
]

RECOMMENDED_SIM_WL_FROM_UM = 0.38
RECOMMENDED_SIM_WL_TO_UM = 0.78
