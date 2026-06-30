"""Shared Streamlit page CSS for filmstack simulation and optimization pages."""

from __future__ import annotations

import streamlit as st

from filmstack_simulation.page_widgets import (
    OPT_FORMULA_STACK_HEIGHT_PX,
    SIM_FORMULA_STACK_HEIGHT_PX,
)

_SIM_CSS = """        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="block-container"],
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="stAppViewBlockContainer"] {{
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            max-width: none !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker) [data-testid="stColumn"] {{
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
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }}
        .__PREFIX__-panel-head-link {{
            text-transform: none;
            letter-spacing: normal;
            font-weight: 500;
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
        .__PREFIX__-input-row-marker,
        .__PREFIX__-formula-area-marker,
        .__PREFIX__-params-stack-marker,
        .__PREFIX__-preset-row-marker,
        .__PREFIX__-pol-row-marker,
        .__PREFIX__-params-spacer-marker,
        .__PREFIX__-page-marker {{
            display: none;
        }}
        [data-testid="stElementContainer"]:has(> div > .__PREFIX__-formula-area-marker),
        [data-testid="stElementContainer"]:has(> div > .__PREFIX__-params-stack-marker),
        [data-testid="stElementContainer"]:has(> div > .__PREFIX__-preset-row-marker),
        [data-testid="stElementContainer"]:has(> div > .__PREFIX__-pol-row-marker) {{
            display: none !important;
            margin: 0 !important;
            padding: 0 !important;
            min-height: 0 !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="stColumn"]:has(.__PREFIX__-formula-area-marker) [data-testid="stTextArea"] {{
            flex: 0 0 auto !important;
            height: auto !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="stColumn"]:has(.__PREFIX__-formula-area-marker) [data-testid="stTextAreaRootElement"] {{
            flex-grow: 0 !important;
            height: __FORMULA_STACK_HEIGHT__ !important;
            min-height: __FORMULA_STACK_HEIGHT__ !important;
            max-height: __FORMULA_STACK_HEIGHT__ !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="stColumn"]:has(.__PREFIX__-formula-area-marker) [data-testid="stTextArea"] textarea {{
            height: 100% !important;
            min-height: 0 !important;
            max-height: 100% !important;
            flex-grow: 1 !important;
            resize: none !important;
            box-sizing: border-box !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="stHorizontalBlock"]:has(.__PREFIX__-params-stack-marker) {{
            align-items: flex-start !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker)
            > div > [data-testid="stVerticalBlock"] {{
            display: flex !important;
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: var(--space-3) !important;
            justify-content: flex-start !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker)
            > div > [data-testid="stVerticalBlock"]
            > [data-testid="stElementContainer"]:has([data-testid="stButton"]) {{
            width: 100% !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker)
            > div > [data-testid="stVerticalBlock"]
            > [data-testid="stElementContainer"]:has([data-testid="stHorizontalBlock"]) {{
            width: 100% !important;
            max-width: 100% !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker)
            [data-testid="stHorizontalBlock"] {{
            width: 100% !important;
            min-height: 2.25rem;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker)
            [data-testid="stSelectbox"],
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker)
            [data-testid="stButton"] {{
            width: 100% !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker)
            [data-testid="stSelectbox"] [data-testid="stWidgetLabel"] {{
            display: none !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker)
            [data-testid="stSelectbox"] > div > div,
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker)
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
            [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker)
            [data-testid="stSelectbox"] div[data-baseweb="select"] span,
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker)
            [data-testid="stSelectbox"] div[data-baseweb="select"] input {{
            font-family: var(--font-ui) !important;
            font-size: 0.875rem !important;
            font-weight: 400 !important;
            line-height: 1.25rem !important;
            color: var(--color-text) !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker)
            [data-testid="stSelectbox"] > div > div:focus-within {{
            border-color: var(--color-accent-material) !important;
            outline: none;
            box-shadow: 0 0 0 1px var(--color-accent-material) !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker)
            > div > [data-testid="stVerticalBlock"]
            > [data-testid="stElementContainer"]:has([data-testid="stHorizontalBlock"]) {{
            width: 100% !important;
            max-width: 100% !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker)
            [data-testid="stHorizontalBlock"] [data-testid="stSelectbox"] {{
            width: 100% !important;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker)
            [data-testid="stTextInput"] input {{
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
            [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker)
            [data-testid="stTextInput"] input:focus {{
            border-color: var(--color-accent-material) !important;
            outline: none;
            box-shadow: 0 0 0 1px var(--color-accent-material) !important;
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
            [data-testid="stColumn"]:not(:has(.__PREFIX__-params-stack-marker))
            [data-testid="stSelectbox"] > div > div,
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="stColumn"]:not(:has(.__PREFIX__-params-stack-marker))
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
            [data-testid="stColumn"]:not(:has(.__PREFIX__-params-stack-marker))
            [data-testid="stSelectbox"] div[data-baseweb="select"] span,
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="stColumn"]:not(:has(.__PREFIX__-params-stack-marker))
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
            [data-testid="stColumn"]:not(:has(.__PREFIX__-params-stack-marker))
            [data-testid="stSelectbox"] > div > div:focus-within {{
            border-color: var(--color-accent-material) !important;
            outline: none;
            box-shadow: 0 0 0 1px var(--color-accent-material) !important;
        }}
        /* Hide number-input +/- steppers */
        .stApp:has(.__PREFIX__-page-marker) [data-testid="stNumberInputStepUp"],
        .stApp:has(.__PREFIX__-page-marker) [data-testid="stNumberInputStepDown"],
        .stApp:has(.__PREFIX__-page-marker) [data-testid="stNumberInputIcon"],
        .stApp:has(.__PREFIX__-page-marker) [data-testid="stNumberInputContainer"] button,
        [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker) [data-testid="stNumberInputStepUp"],
        [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker) [data-testid="stNumberInputStepDown"],
        [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker) [data-testid="stNumberInputContainer"] button {{
            display: none !important;
            width: 0 !important;
            min-width: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            overflow: hidden !important;
            pointer-events: none !important;
        }}
        .stApp:has(.__PREFIX__-page-marker) [data-testid="stNumberInputContainer"],
        [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker) [data-testid="stNumberInputContainer"] {{
            gap: 0 !important;
        }}
        .stApp:has(.__PREFIX__-page-marker) [data-testid="stNumberInput"] > div,
        [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker) [data-testid="stNumberInput"] > div {{
            gap: 0 !important;
        }}
        .stApp:has(.__PREFIX__-page-marker) [data-testid="stNumberInput"] input,
        [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker) [data-testid="stNumberInput"] input {{
            width: 100% !important;
            -moz-appearance: textfield !important;
        }}
        .stApp:has(.__PREFIX__-page-marker) [data-testid="stNumberInput"] input::-webkit-outer-spin-button,
        .stApp:has(.__PREFIX__-page-marker) [data-testid="stNumberInput"] input::-webkit-inner-spin-button,
        [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker) [data-testid="stNumberInput"] input::-webkit-outer-spin-button,
        [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker) [data-testid="stNumberInput"] input::-webkit-inner-spin-button {{
            -webkit-appearance: none !important;
            margin: 0 !important;
        }}
        .stApp:has(.__PREFIX__-page-marker) [data-baseweb="numberinput"] button,
        [data-testid="stColumn"]:has(.__PREFIX__-params-stack-marker) [data-baseweb="numberinput"] button {{
            display: none !important;
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


_OPT_EXTRA = """        .__PREFIX__-layer-table-marker {{
            display: none;
        }}
        /* Layer table: block header clicks (fixed rows allow sort; overlay disables it) */
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="stVerticalBlock"]:has(.__PREFIX__-layer-table-marker)
            [data-testid="stDataEditor"] > div {{
            position: relative;
        }}
        [data-testid="stAppViewContainer"]:has(.__PREFIX__-page-marker)
            [data-testid="stVerticalBlock"]:has(.__PREFIX__-layer-table-marker)
            [data-testid="stDataEditor"] > div::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2.5rem;
            z-index: 5;
            pointer-events: auto;
            cursor: default;
        }}
        .__PREFIX__-section-label {{
            font-family: var(--font-ui);
            font-size: 0.9375rem;
            font-weight: 600;
            color: var(--color-text);
            margin: var(--space-3) 0 var(--space-2) 0;
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }}
"""

_OPT_CSS = _SIM_CSS + _OPT_EXTRA


def _apply_prefix(css: str, *, prefix: str) -> str:
    return css.replace("__PREFIX__", prefix)


def inject_page_styles(*, prefix: str, tokens_css: str, variant: str) -> None:
    """Inject design tokens plus page layout CSS (variant: ``sim`` or ``opt``)."""
    template = _SIM_CSS if variant == "sim" else _OPT_CSS
    height_px = (
        SIM_FORMULA_STACK_HEIGHT_PX if variant == "sim" else OPT_FORMULA_STACK_HEIGHT_PX
    )
    body = _apply_prefix(template, prefix=prefix).replace(
        "__FORMULA_STACK_HEIGHT__",
        f"{height_px}px",
    )
    st.markdown(
        f"""
        <div class="{prefix}-page-marker"></div>
        <style>
        {tokens_css}
        {body}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_filmstack_sim_styles(tokens_css: str) -> None:
    inject_page_styles(prefix="fs-sim", tokens_css=tokens_css, variant="sim")


def inject_filmstack_opt_styles(tokens_css: str) -> None:
    inject_page_styles(prefix="fs-opt", tokens_css=tokens_css, variant="opt")
