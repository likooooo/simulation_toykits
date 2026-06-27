"""Shared help text and documentation URLs for filmstack pages."""

FORMULA_DOCS_URL = (
    "https://github.com/likooooo/simulation_toykits/blob/main/docs/filmstack_formula_usage.md"
)

FORMULA_HELP_TEXT = (
    "格式：Material thickness_um [n k]；(…)^N 周期。\n"
    "不在工作区的材料必须写 n k。详见使用说明。"
)

WL_RANGE_LABEL = "仿真波长范围\u00a0(μm)"

PARAMS_STALE_INFO = "参数已修改，请重新点击「仿真」。"

POLARIZATION_STALE_INFO = "偏振已更改，请重新点击「仿真」。"

PRESET_STALE_INFO = "预设膜系已更改，请重新点击「仿真」。"

OPTIMIZED_FORMULA_HELP_TEXT = (
    "完成一次 Freehand 优化后将在此显示最新膜系指令。\n"
    "复制到 Filmstack Simulation 页面可查看更详细的性能指标。"
)

FREEHAND_CHART_HELP_TEXT = (
    "只有视图范围内的波长参与优化；可用 Alt+拖动缩放视图。\n"
    "为节省计算资源，优化器默认使用较宽松的收敛参数。"
)
