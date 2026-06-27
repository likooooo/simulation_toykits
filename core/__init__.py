"""计算逻辑层；beams / sturm_liouville 等按需懒加载（依赖 simulation.so）。"""

from common import get_nk_at_wavelength

_LAZY_MODULES = {
    "compute_plane_wave": "core.beams",
    "compute_quadratic_wave": "core.beams",
    "compute_spherical_wave": "core.beams",
    "compute_flat_top_rectangular": "core.beams",
    "compute_flat_top_circular": "core.beams",
    "compute_hermite_gaussian": "core.beams",
    "compute_laguerre_gaussian": "core.beams",
    "show_complex_plot": "core.beams_plot",
}


def __getattr__(name):
    if name in _LAZY_MODULES:
        import importlib
        mod = importlib.import_module(_LAZY_MODULES[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "get_nk_at_wavelength",
    "compute_plane_wave",
    "compute_quadratic_wave",
    "compute_spherical_wave",
    "compute_flat_top_rectangular",
    "compute_flat_top_circular",
    "compute_hermite_gaussian",
    "compute_laguerre_gaussian",
    "show_complex_plot",
]
