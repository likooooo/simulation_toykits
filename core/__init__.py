# core: 计算逻辑层，所有函数通过参数传参，不依赖全局变量，可在其他项目中复用。
# 依赖 assets.simulation 的模块在无 assets 时可选不加载，便于 CI 等环境只跑不依赖 assets 的测试。

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

# 以下依赖 assets.simulation，无 assets 时（如 CI 私有子模块未拉取）跳过，不阻塞 core 包加载
try:
    from core.fresnel import build_tmm_layers, compute_RT, get_r_t
    from core.films import compute_fresnel_and_filmstack, FresnelFilmstackResult
    from core.spectral import (
        compute_angle_vs_RT_figures,
        compute_wavelength_vs_RT_figures,
        compute_TE_TM_wavelength_angle_figures,
        build_nk_map_for_wavelengths,
    )
    _HAS_ASSETS = True
except (ImportError, ModuleNotFoundError):
    _HAS_ASSETS = False
    build_tmm_layers = None
    compute_RT = None
    get_r_t = None
    compute_fresnel_and_filmstack = None
    FresnelFilmstackResult = None
    compute_angle_vs_RT_figures = None
    compute_wavelength_vs_RT_figures = None
    compute_TE_TM_wavelength_angle_figures = None
    build_nk_map_for_wavelengths = None

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
]
