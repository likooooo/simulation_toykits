"""Figure display helpers for Streamlit."""

from __future__ import annotations

import base64
import io
from typing import Any

import streamlit as st

PAGE_WIDTH_RATIO = 0.8
_PLOTLY_CHART_CONFIG = {"displayModeBar": True, "displaylogo": False}


def show_figure(fig: Any, *, dpi: int = 100) -> None:
    """Render matplotlib Figure at fixed page width ratio."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    ratio_pct = int(PAGE_WIDTH_RATIO * 100)
    st.markdown(
        f'<div style="width:{ratio_pct}%; margin:0 auto;">'
        f'<img src="data:image/png;base64,{b64}" style="width:100%; height:auto; display:block;"/>'
        "</div>",
        unsafe_allow_html=True,
    )


def show_plotly_figure(fig: Any, *, key: str | None = None) -> None:
    """Render Plotly figure full width (Simulation Database chart style)."""
    st.plotly_chart(
        fig,
        width="stretch",
        key=key,
        config=_PLOTLY_CHART_CONFIG,
    )
