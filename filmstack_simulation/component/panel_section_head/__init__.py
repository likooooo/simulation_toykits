"""Panel section header Streamlit component (Simulation Database style)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import streamlit.components.v1 as components

_COMPONENT_DIR = str(Path(__file__).resolve().parent / "frontend")
_panel_section_head = components.declare_component("panel_section_head", path=_COMPONENT_DIR)


def _default_key(title: str, css_prefix: str) -> str:
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "_", title.strip()).strip("_")[:48]
    return f"{css_prefix}_section_{slug or 'head'}"


def panel_section_head(
    title: str,
    *,
    help_text: str | None = None,
    help_url: str | None = None,
    align: Literal["left", "right"] = "left",
    key: str | None = None,
    css_prefix: str = "panel",
    tokens_css: str = "",
) -> None:
    """Render a section title with optional ``?`` help tooltip and docs link."""
    _panel_section_head(
        title=title,
        help_text=help_text or "",
        help_url=help_url or "",
        align=align,
        key=key or _default_key(title, css_prefix),
        tokens_css=tokens_css,
        default=None,
    )
