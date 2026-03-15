"""计算逻辑层；fresnel/films/spectral 按需懒加载（依赖 simulation.so）。"""

from core.materials import get_nk_at_wavelength, with_nk_columns
from core.formula import parse_formula, parse_formula_v1
from core.filmstack_viz import (
    MockMaterial,
    MockFilm,
    calculate_angles,
    plot_periodic_structure,
    nk_to_color,
    build_layers_for_visualization,
    plot_tmm_filmstack,
)
from core.material_index import load_material_index_cached

_LAZY_MODULES = {
    "build_tmm_layers": "core.fresnel",
    "compute_RT": "core.fresnel",
    "get_r_t": "core.fresnel",
    "compute_fresnel_and_filmstack": "core.films",
    "FresnelFilmstackResult": "core.films",
    "compute_angle_vs_RT_figures": "core.spectral",
    "compute_wavelength_vs_RT_figures": "core.spectral",
    "compute_TE_TM_wavelength_angle_figures": "core.spectral",
    "build_nk_map_for_wavelengths": "core.spectral",
    "compute_plane_wave": "core.beams",
    "compute_quadratic_wave": "core.beams",
    "compute_spherical_wave": "core.beams",
    "compute_flat_top_rectangular": "core.beams",
    "compute_flat_top_circular": "core.beams",
    "compute_hermite_gaussian": "core.beams",
    "compute_laguerre_gaussian": "core.beams",
    "show_complex_plot": "core.beams_plot",
    "load_mat_v7": "core.sturm_liouville",
    "sl_formula_markdown": "core.sturm_liouville",
    "run_sturm_liouville": "core.sturm_liouville",
    "run_time_dependent_sturm_liouville": "core.sturm_liouville",
    "build_result_mat": "core.sturm_liouville",
    "plot_result_and_error": "core.sturm_liouville",
}


def __getattr__(name):
    if name in _LAZY_MODULES:
        import importlib
        mod = importlib.import_module(_LAZY_MODULES[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "get_nk_at_wavelength",
    "with_nk_columns",
    "parse_formula",
    "parse_formula_v1",
    "MockMaterial",
    "MockFilm",
    "calculate_angles",
    "plot_periodic_structure",
    "nk_to_color",
    "build_layers_for_visualization",
    "plot_tmm_filmstack",
    "load_material_index_cached",
    "build_tmm_layers",
    "compute_RT",
    "get_r_t",
    "compute_fresnel_and_filmstack",
    "FresnelFilmstackResult",
    "compute_angle_vs_RT_figures",
    "compute_wavelength_vs_RT_figures",
    "compute_TE_TM_wavelength_angle_figures",
    "build_nk_map_for_wavelengths",
    "compute_plane_wave",
    "compute_quadratic_wave",
    "compute_spherical_wave",
    "compute_flat_top_rectangular",
    "compute_flat_top_circular",
    "compute_hermite_gaussian",
    "compute_laguerre_gaussian",
    "show_complex_plot",
    "load_mat_v7",
    "sl_formula_markdown",
    "run_sturm_liouville",
    "run_time_dependent_sturm_liouville",
    "build_result_mat",
    "plot_result_and_error",
]
