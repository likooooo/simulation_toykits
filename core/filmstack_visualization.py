"""
膜系可视化：从 core.filmstack_viz 重新导出，保持向后兼容。
新代码请直接从 core 或 core.filmstack_viz 导入。
"""

from core.filmstack_viz import (
    MockMaterial,
    MockFilm,
    calculate_angles,
    plot_periodic_structure,
    nk_to_color,
    build_layers_for_visualization,
    plot_tmm_filmstack,
)

__all__ = [
    "MockMaterial",
    "MockFilm",
    "calculate_angles",
    "plot_periodic_structure",
    "nk_to_color",
    "build_layers_for_visualization",
    "plot_tmm_filmstack",
]
