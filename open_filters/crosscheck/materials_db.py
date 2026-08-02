"""Material token paths under simulation_database of/ subtree."""

from __future__ import annotations

OF_FILMSTACK_MATERIAL_PATHS: dict[str, list[str]] = {
    "of_void": ["of", "materials", "void.yml"],
    "of_TiO2": ["of", "materials", "TiO2.yml"],
    "of_SiO2": ["of", "materials", "SiO2.yml"],
    "of_BK7": ["of", "materials", "BK7.yml"],
}

# Materials included in nk alignment tests (regular OpenFilters catalog subset).
OF_ALIGNMENT_MATERIALS: dict[str, list[str]] = {
    "of_void": ["of", "materials", "void.yml"],
    "of_TiO2": ["of", "materials", "TiO2.yml"],
    "of_SiO2": ["of", "materials", "SiO2.yml"],
    "of_BK7": ["of", "materials", "BK7.yml"],
    "of_FusedSilica": ["of", "materials", "FusedSilica.yml"],
}
