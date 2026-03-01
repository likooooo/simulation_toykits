"""
通用绘图：仅依赖数据与绘图库，无 st 依赖。返回 Figure 供调用方显示或保存。
"""

from __future__ import annotations

from typing import List, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import plotly.graph_objects as go

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    go = None
    _HAS_PLOTLY = False


def build_nk_curve_figure(
    wavelengths: Union[List[float], np.ndarray],
    n_vals: Union[List[float], np.ndarray],
    k_vals: Union[List[float], np.ndarray],
    title: str = "n, k",
) -> "go.Figure":
    """
    绘制材料 n、k 随波长曲线，返回 Plotly Figure。无 st 依赖。

    :param wavelengths: 波长 (μm)
    :param n_vals: 折射率 n
    :param k_vals: 消光系数 k
    :param title: 图标题
    :return: plotly.graph_objects.Figure
    """
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required for build_nk_curve_figure")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(wavelengths),
            y=list(n_vals),
            name="n",
            line=dict(color="#DC3545", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(wavelengths),
            y=list(k_vals),
            name="k",
            line=dict(color="#007BFF", width=2),
        )
    )
    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis_title="Wavelength (μm)",
        yaxis_title="n, k",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )
    return fig
