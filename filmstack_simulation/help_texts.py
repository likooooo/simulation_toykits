"""Shared help text and documentation URLs for filmstack pages."""

from __future__ import annotations

import os
from pathlib import Path

from filmstack_simulation.simulation import DEFAULT_N_ANG, DEFAULT_N_WL

FORMULA_DOCS_URL = "/filmstack-formula-usage"

FORMULA_HELP_TEXT = (
    "格式：Material thickness_um [n k]；(…)^N 周期；Maxwell-Garnett：[q inclusion host [n [k]]] 厚度_um。\n"
    "Database 材料直接使用名称；不在工作区的材料须写 inline n k。\n"
    "预设膜厚已在设计波长下离线展开为字面 μm，切换「仿真波长范围」不会重算；自定义公式为字面厚度。详见使用说明。"
)

WL_RANGE_LABEL = "仿真波长范围\u00a0(μm)"

PARAMS_STALE_INFO = "参数已修改，请重新点击「仿真」。"

POLARIZATION_STALE_INFO = "偏振已更改，请重新点击「仿真」。"

PRESET_STALE_INFO = "膜系配置已更改，请重新点击「仿真」。"

OPTIMIZED_FORMULA_HELP_TEXT = (
    "完成一次 Freehand 优化后将在此显示最新膜系指令。\n"
    "复制到 Filmstack Simulation 页面可查看更详细的性能指标。"
)

FREEHAND_CHART_HELP_TEXT = (
    "已编辑的 R/T/A 曲线中，仅当前视图可见波长参与优化（默认等于全范围；Alt+拖动可缩放视图）。\n"
    "为节省计算资源，优化器默认使用较宽松的收敛参数。"
)

NK_CURVE_HELP_TEXT = (
    f"膜系各材料在仿真波长范围（右侧参数区「{WL_RANGE_LABEL}」）内的折射率 n 与消光系数 k。\n"
    "Database 材料使用 Simulation Database 色散模型；公式 inline n/k 为常数折射率。"
)

SPECTRAL_MAP_HELP_TEXT = (
    "波长–角度二维谱图：R、T、Ψ、Δ。\n"
    "R/T 随参数区「偏振」选择变化；Ψ/Δ 由 s/p 反射系数比计算，与偏振选择无关。\n"
    f"角度网格采样点数: {DEFAULT_N_ANG}。\n"
    f"波长网格采样点数: {DEFAULT_N_WL}。"
)

SLICE_AT_WL_HELP_TEXT = (
    "固定目标波长，R/T/Ψ/Δ 随入射角变化的一维曲线。\n"
    f"角度网格采样点数: {DEFAULT_N_ANG}。"
)

SLICE_AT_ANGLE_HELP_TEXT = (
    "固定目标角度，R/T/Ψ/Δ 随波长变化的一维曲线。\n"
    f"波长网格采样点数: {DEFAULT_N_WL}。\n"
    f"奈奎斯特截止频率：{(DEFAULT_N_WL-1)/2}/Δλ。\n"
    "为节省计算资源，使用较为稀疏的采样点，高频成分丢失导致部分基准测试无法复现。"
)

FREESNEL_COMPARE_PAGE_URL = "/fs-baseline-compare"
FREESNEL_COMPARE_ARTIFACT_REL = Path("assets") / "fs_compare" / "fs_baseline_vs_toykits.html"


def fs_compare_artifact_path() -> Path | None:
    artifacts = os.environ.get("SIMULATION_ARTIFACTS_DIR", "").strip()
    if not artifacts:
        return None
    return Path(artifacts) / FREESNEL_COMPARE_ARTIFACT_REL


def fs_compare_artifact_available() -> bool:
    path = fs_compare_artifact_path()
    return path is not None and path.is_file()
