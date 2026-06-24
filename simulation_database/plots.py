"""Plotly figures for material nk and spectrum curves."""

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

PLOTLY_CHART_CONFIG = {"displayModeBar": True, "displaylogo": False}

COLORS = {
    "text": "#111827",
    "text_secondary": "#6B7280",
    "grid": "#E5E7EB",
    "material_n": "#2563EB",
    "material_k": "#DC2626",
    "spectrum": "#D97706",
    "bg": "#FFFFFF",
}


def apply_plotly_theme(fig: "go.Figure", height: int = 420) -> "go.Figure":
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=COLORS["text"]),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        height=height,
        margin=dict(l=48, r=24, t=48, b=48),
        hovermode="x unified",
        xaxis=dict(
            showgrid=True,
            gridcolor=COLORS["grid"],
            linecolor=COLORS["grid"],
            title_font=dict(size=12),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLORS["grid"],
            linecolor=COLORS["grid"],
            title_font=dict(size=12),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
    )
    if getattr(fig.layout, "yaxis2", None) is not None:
        fig.update_layout(
            margin=dict(l=48, r=56, t=48, b=48),
            yaxis2=dict(
                showgrid=False,
                linecolor=COLORS["grid"],
                title_font=dict(size=12),
                tickfont=dict(size=11),
            ),
        )
    return fig


_SIM_WL_MARKER_LINE = dict(color="rgba(107, 114, 128, 0.85)", width=1.5, dash="dash")


def _apply_sim_wl_markers(
    fig: "go.Figure",
    wavelengths: Union[List[float], np.ndarray],
    sim_wl_from: float | None,
    sim_wl_to: float | None,
) -> None:
    wl_arr = np.asarray(wavelengths, dtype=float)
    if wl_arr.size == 0:
        return
    data_min = float(np.min(wl_arr))
    data_max = float(np.max(wl_arr))
    x_min = data_min
    x_max = data_max
    markers: list[float] = []
    if sim_wl_from is not None:
        markers.append(float(sim_wl_from))
    if sim_wl_to is not None:
        markers.append(float(sim_wl_to))
    if markers:
        x_min = min(x_min, min(markers))
        x_max = max(x_max, max(markers))
    fig.update_xaxes(range=[x_min, x_max])
    for x_val in markers:
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=x_val,
            x1=x_val,
            y0=0,
            y1=1,
            line=dict(_SIM_WL_MARKER_LINE),
            layer="below",
        )


def build_nk_curve_figure(
    wavelengths: Union[List[float], np.ndarray],
    n_vals: Union[List[float], np.ndarray],
    k_vals: Union[List[float], np.ndarray],
    title: str | None = None,
    sim_wl_from: float | None = None,
    sim_wl_to: float | None = None,
    height: int = 500,
) -> "go.Figure":
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required for build_nk_curve_figure")
    n_arr = np.asarray(n_vals, dtype=float)
    k_arr = np.asarray(k_vals, dtype=float)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(wavelengths),
            y=list(n_arr),
            name="n",
            line=dict(color=COLORS["material_n"], width=2),
            yaxis="y",
            hovertemplate="n=%{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(wavelengths),
            y=list(k_arr),
            name="k",
            line=dict(color=COLORS["material_k"], width=2),
            yaxis="y2",
            hovertemplate="k=%{y}<extra></extra>",
        )
    )
    n_min, n_max = float(np.min(n_arr)), float(np.max(n_arr))
    k_min, k_max = float(np.min(k_arr)), float(np.max(k_arr))
    n_pad = max((n_max - n_min) * 0.08, 1e-6) if n_max > n_min else max(abs(n_max) * 0.08, 1e-6)
    k_pad = max((k_max - k_min) * 0.08, 1e-9) if k_max > k_min else max(abs(k_max) * 0.08, 1e-9)
    layout: dict = {
        "xaxis_title": "Wavelength (μm)",
        "hovermode": "x unified",
        "template": "plotly_white",
        "yaxis": dict(
            title="n",
            side="left",
            range=[n_min - n_pad, n_max + n_pad],
            showgrid=True,
        ),
        "yaxis2": dict(
            title="k",
            side="right",
            overlaying="y",
            anchor="x",
            range=[max(0.0, k_min - k_pad), k_max + k_pad],
            showgrid=False,
        ),
    }
    if title:
        layout["title"] = f"<b>{title}</b>"
    fig.update_layout(**layout)
    _apply_sim_wl_markers(fig, wavelengths, sim_wl_from, sim_wl_to)
    return apply_plotly_theme(fig, height=height)


def build_spectrum_curve_figure(
    wavelengths_um: Union[List[float], np.ndarray],
    values: Union[List[float], np.ndarray],
    title: str | None = None,
    sim_wl_from: float | None = None,
    sim_wl_to: float | None = None,
    height: int = 500,
) -> "go.Figure":
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required for build_spectrum_curve_figure")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(wavelengths_um),
            y=list(values),
            name="spectrum",
            line=dict(color=COLORS["spectrum"], width=2),
        )
    )
    layout: dict = {
        "xaxis_title": "Wavelength (μm)",
        "yaxis_title": "Intensity",
        "hovermode": "x unified",
        "template": "plotly_white",
    }
    if title:
        layout["title"] = f"<b>{title}</b>"
    fig.update_layout(**layout)
    _apply_sim_wl_markers(fig, wavelengths_um, sim_wl_from, sim_wl_to)
    return apply_plotly_theme(fig, height=height)
