"""Global UI theme injection for Simulation Toykits."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_TOKENS_PATH = Path(__file__).resolve().parent / "design_tokens.css"


def inject_global_styles() -> None:
    """Inject design tokens and Streamlit layout overrides once per session."""
    if st.session_state.get("_ui_theme_injected"):
        return
    tokens = _TOKENS_PATH.read_text(encoding="utf-8")
    st.markdown(
        f"""
        <style>
        {tokens}
        html {{ overflow-y: scroll !important; }}
        [data-testid="stAppViewBlockContainer"] {{
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            max-width: none !important;
        }}
        [data-testid="column"] {{
            min-width: 0 !important;
        }}
        .stApp {{
            background: var(--color-bg) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["_ui_theme_injected"] = True
