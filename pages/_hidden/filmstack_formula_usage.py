"""Hidden viewer for Filmstack formula usage documentation."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from common import show_markdown_file

st.set_page_config(page_title="多层膜构建指令", layout="wide")

_DOC_PATH = Path(__file__).resolve().parents[2] / "docs" / "filmstack_formula_usage.md"
show_markdown_file(_DOC_PATH)
