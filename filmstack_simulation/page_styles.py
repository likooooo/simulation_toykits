"""Shared Streamlit page CSS for filmstack simulation and optimization pages."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_SIM_CSS = """        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="block-container"],
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="stAppViewBlockContainer"] {{
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            max-width: none !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="column"] {{
            flex: 1 1 0% !important;
            min-width: 0 !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="stHorizontalBlock"] {{
            align-items: flex-start;
            gap: 0.5rem;
        }}
        .__PREFIX__-title {{
            font-family: var(--font-ui);
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--color-text);
            margin: 0 0 var(--space-4) 0;
            line-height: 1.2;
            white-space: nowrap;
        }}
        .__PREFIX__-panel-head {{
            font-family: var(--font-ui);
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--color-text-secondary, #6B7280);
            margin: 0;
            line-height: 2.25rem;
            white-space: nowrap;
            flex-shrink: 0;
        }}
        .__PREFIX__-panel-head a {{
            color: var(--color-accent-material, #2563EB);
            text-decoration: none;
            font-weight: 500;
            text-transform: none;
            letter-spacing: normal;
        }}
        .__PREFIX__-panel-head a:hover {{
            text-decoration: underline;
        }}
        .__PREFIX__-param-left,
        .__PREFIX__-param-right {{
            display: none;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="column"]:has(.__PREFIX__-param-left) > div > [data-testid="stVerticalBlock"] {{
            align-items: flex-start !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="column"]:has(.__PREFIX__-param-right) > div > [data-testid="stVerticalBlock"] {{
            align-items: flex-end !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="column"]:has(.__PREFIX__-param-left)
            > div > [data-testid="stVerticalBlock"]
            > [data-testid="element-container"]:has([data-testid="stHorizontalBlock"]) {{
            width: fit-content !important;
            max-width: 100% !important;
            align-self: flex-start !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="column"]:has(.__PREFIX__-param-right)
            > div > [data-testid="stVerticalBlock"]
            > [data-testid="element-container"]:has([data-testid="stHorizontalBlock"]) {{
            width: fit-content !important;
            max-width: 100% !important;
            align-self: flex-end !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="column"]:has(.__PREFIX__-param-left) [data-testid="stHorizontalBlock"] {{
            width: fit-content !important;
            max-width: 100% !important;
            justify-content: flex-start !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="column"]:has(.__PREFIX__-param-right) [data-testid="stHorizontalBlock"] {{
            width: fit-content !important;
            max-width: 100% !important;
            justify-content: flex-end !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="column"]:has(.__PREFIX__-param-right) .__PREFIX__-panel-head {{
            text-align: right;
        }}
        /* Build row: formula (5) + preset/button stack (1), height 120px */
        .__PREFIX__-build-row-marker,
        .__PREFIX__-build-right-marker {{
            display: none;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="stHorizontalBlock"]:has(.__PREFIX__-build-row-marker)
            + [data-testid="stHorizontalBlock"] {{
            align-items: stretch !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="column"]:has(.__PREFIX__-build-right-marker)
            > div > [data-testid="stVerticalBlock"] {{
            display: flex !important;
            flex-direction: column !important;
            min-height: 132px !important;
            gap: var(--space-2) !important;
            justify-content: space-between !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="column"]:has(.__PREFIX__-build-right-marker)
            [data-testid="stSelectbox"],
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="column"]:has(.__PREFIX__-build-right-marker)
            [data-testid="stButton"] {{
            width: 100% !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="column"]:has(.__PREFIX__-build-right-marker)
            [data-testid="stSelectbox"] [data-testid="stWidgetLabel"] {{
            display: none !important;
        }}
        /* Unified text / number / select / textarea inputs */
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="stNumberInput"],
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="stTextInput"],
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="stTextArea"],
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="stSelectbox"] {{
            margin-bottom: 0;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="stNumberInput"] input,
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="stTextInput"] input,
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="stTextArea"] textarea,
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="column"]:not(:has(.__PREFIX__-build-right-marker))
            [data-testid="stSelectbox"] > div > div,
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="column"]:not(:has(.__PREFIX__-build-right-marker))
            [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
            font-family: var(--font-ui) !important;
            font-size: 0.875rem !important;
            font-weight: 400 !important;
            line-height: 1.25rem !important;
            color: var(--color-text) !important;
            background-color: var(--color-surface) !important;
            border: 1px solid var(--color-border) !important;
            border-radius: var(--radius-sm) !important;
            padding: 0.35rem 0.5rem !important;
            min-height: 2rem;
            box-shadow: none !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="column"]:not(:has(.__PREFIX__-build-right-marker))
            [data-testid="stSelectbox"] div[data-baseweb="select"] span,
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="column"]:not(:has(.__PREFIX__-build-right-marker))
            [data-testid="stSelectbox"] div[data-baseweb="select"] input {{
            font-family: var(--font-ui) !important;
            font-size: 0.875rem !important;
            font-weight: 400 !important;
            line-height: 1.25rem !important;
            color: var(--color-text) !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="stNumberInput"] input:focus,
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="stTextInput"] input:focus,
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="stTextArea"] textarea:focus,
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="column"]:not(:has(.__PREFIX__-build-right-marker))
            [data-testid="stSelectbox"] > div > div:focus-within {{
            border-color: var(--color-accent-material) !important;
            outline: none;
            box-shadow: 0 0 0 1px var(--color-accent-material) !important;
        }}
        /* Hide number-input +/- steppers */
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="stNumberInputStepUp"],
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="stNumberInputStepDown"],
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="stNumberInputIcon"] {{
            display: none !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="stNumberInput"] > div {{
            gap: 0 !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) hr {{
            margin: var(--space-4) 0;
            border-color: var(--color-border, #E5E7EB);
        }}
        .__PREFIX__-section-label {{
            font-family: var(--font-ui);
            font-size: 0.9375rem;
            font-weight: 600;
            color: var(--color-text);
            margin: 0 0 var(--space-2) 0;
            white-space: nowrap;
        }}
        .__PREFIX__-chart-title {{
            font-family: var(--font-ui);
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--color-text-secondary);
            margin: var(--space-3) 0 var(--space-2) 0;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            white-space: nowrap;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) .__PREFIX__-action-row [data-testid="stHorizontalBlock"] {{
            margin-top: var(--space-1);
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) .__PREFIX__-action-row [data-testid="column"]:last-child {{
            flex: 0 0 auto !important;
            min-width: 7rem !important;
            max-width: 8rem !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="stButton"] > button {{
            font-family: var(--font-ui) !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            color: var(--color-text) !important;
            background-color: var(--color-surface) !important;
            border: 1px solid var(--color-border-strong) !important;
            border-radius: var(--radius-sm) !important;
            padding: 0.35rem 0.75rem !important;
            min-height: 2rem;
            box-shadow: var(--shadow-sm) !important;
            transition: background-color var(--transition-fast), border-color var(--transition-fast);
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="stButton"] > button:hover {{
            background-color: var(--color-hover) !important;
            border-color: var(--color-border-strong) !important;
            color: var(--color-text) !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="stButton"] > button:active {{
            background-color: var(--color-border) !important;
        }}
"""

_OPT_CSS = """        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="block-container"],
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="stAppViewBlockContainer"] {{
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            max-width: none !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="column"] {{
            flex: 1 1 0% !important;
            min-width: 0 !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="stHorizontalBlock"] {{
            align-items: flex-start;
            gap: 0.5rem;
        }}
        .__PREFIX__-build-row-marker,
        .__PREFIX__-build-right-marker,
        .__PREFIX__-param-right-marker {{
            display: none;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="stHorizontalBlock"]:has(.__PREFIX__-build-row-marker)
            + [data-testid="stHorizontalBlock"] {{
            align-items: stretch !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="column"]:has(.__PREFIX__-build-right-marker)
            > div > [data-testid="stVerticalBlock"] {{
            display: flex !important;
            flex-direction: column !important;
            min-height: 132px !important;
            gap: var(--space-2) !important;
            justify-content: space-between !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="column"]:has(.__PREFIX__-build-right-marker)
            [data-testid="stSelectbox"],
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="column"]:has(.__PREFIX__-build-right-marker)
            [data-testid="stButton"] {{
            width: 100% !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="column"]:has(.__PREFIX__-build-right-marker)
            [data-testid="stSelectbox"] [data-testid="stWidgetLabel"] {{
            display: none !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="column"]:has(.__PREFIX__-param-right-marker)
            > div > [data-testid="stVerticalBlock"] {{
            align-items: flex-end !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="column"]:has(.__PREFIX__-param-right-marker)
            > div > [data-testid="stVerticalBlock"]
            > [data-testid="element-container"]:has([data-testid="stHorizontalBlock"]) {{
            width: fit-content !important;
            max-width: 100% !important;
            align-self: flex-end !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="column"]:has(.__PREFIX__-param-right-marker) [data-testid="stHorizontalBlock"] {{
            width: fit-content !important;
            max-width: 100% !important;
            justify-content: flex-end !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="column"]:has(.__PREFIX__-param-right-marker) .__PREFIX__-panel-head {{
            text-align: right;
        }}
        .__PREFIX__-title {{
            font-family: var(--font-ui);
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--color-text);
            margin: 0 0 var(--space-4) 0;
        }}
        .__PREFIX__-panel-head {{
            font-family: var(--font-ui);
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--color-text-secondary, #6B7280);
            margin: 0;
            line-height: 2.25rem;
        }}
        .__PREFIX__-section-label {{
            font-family: var(--font-ui);
            font-size: 0.9375rem;
            font-weight: 600;
            color: var(--color-text);
            margin: var(--space-3) 0 var(--space-2) 0;
        }}
"""


def _apply_prefix(css: str, *, prefix: str) -> str:
    return css.replace("__PREFIX__", prefix)


def inject_page_styles(*, prefix: str, tokens_path: Path, variant: str) -> None:
    """Inject design tokens plus page layout CSS (variant: ``sim`` or ``opt``)."""
    tokens = tokens_path.read_text(encoding="utf-8")
    template = _SIM_CSS if variant == "sim" else _OPT_CSS
    body = _apply_prefix(template, prefix=prefix)
    st.markdown(
        f"""
        <style>
        {tokens}
        {body}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_filmstack_sim_styles(tokens_path: Path) -> None:
    inject_page_styles(prefix="fs-sim", tokens_path=tokens_path, variant="sim")


def inject_filmstack_opt_styles(tokens_path: Path) -> None:
    inject_page_styles(prefix="fs-opt", tokens_path=tokens_path, variant="opt")
