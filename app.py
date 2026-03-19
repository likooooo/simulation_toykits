import streamlit as st
from pathlib import Path
import re

from core import simulation_loader
simulation_loader.ensure_artifacts_on_path()

st.markdown(
    """
    <style>
    /* 1. 强制滚动条始终显示，从根源消除宽度变化触发点 */
    html {
        overflow-y: scroll !important;
    }

    /* 2. 锁定 st.columns 的列宽计算 */
    /* 我们针对所有的 column 容器，强制取消 flex-basis 的动态调整 */
    [data-testid="column"] {
        flex: 1 1 45% !important; /* 强制两列平分，留出冗余空间 */
        min-width: 0 !important;
    }

    /* 3. 核心修复：禁止 Streamlit 的高度自适应监听器导致的水平抖动 */
    [data-testid="stHorizontalBlock"] {
        width: 100% !important;
        display: flex !important;
        flex-wrap: nowrap !important; /* 禁止换行，这是防止抖动的关键 */
        align-items: flex-start !important;
    }

    /* 4. 给主容器留出足够的 padding 缓冲带，吸收滚动条带来的像素波动 */
    [data-testid="stAppViewBlockContainer"] {
        padding-right: 2rem !important;
        padding-left: 2rem !important;
    }

    /* 5. 解决 iframe 环境下图片缩放触发的重算 */
    [data-testid="stAppViewContainer"] img {
        max-width: 100%;
        height: auto;
        object-fit: contain;
    }

    /* 隐藏 CSS 占位空白 */
    .stMarkdown:has(> div > style) { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

def build_navigation_from_dir(pages_dict=None, base_dir="pages", icon="📄", page_order=None):
    if pages_dict is None:
        pages_dict = {}
    base_path = Path(base_dir)
    py_files = sorted(list(base_path.rglob("*.py")))
    group_name = None

    for py_file in py_files:
        relative_path = py_file.relative_to(base_path.parent)
        parts = relative_path.parts
        group_name = parts[0] if len(parts) > 1 else "Others"
        display_name = py_file.stem.replace("_", " ").title()
        display_name = re.sub(r"^\d+\s*", "", display_name)

        page_obj = st.Page(str(py_file), title=display_name, icon=icon)

        if group_name not in pages_dict:
            pages_dict[group_name] = []
        pages_dict[group_name].append(page_obj)

    if page_order is not None and group_name is not None and group_name in pages_dict:
        def _norm(t):
            return t.strip().lower().replace("-", " ").replace("  ", " ")

        order_map = {_norm(title): i for i, title in enumerate(page_order)}
        pages_dict[group_name].sort(
            key=lambda p: order_map.get(_norm(p.title), len(page_order))
        )

    return pages_dict

pages_dict = {}
pages_dict[""] = [st.Page("pages/main.py", title="home", icon="🏠", default=True)]
build_navigation_from_dir(
    pages_dict,
    "pages/fresnel_caculator",
    icon="📊",
    page_order=[
        "Material Database",
        "Films",
        "Spectral Curve",
        "Angular Spectral Map",
        "Diffraction Angle",
    ],
)
build_navigation_from_dir(
    pages_dict,
    "pages/beams_caculator",
    icon="🎯",
    page_order=[
        "Plane Wave",
        "Quadratic Wave",
        "Spherical Wave",
        "Flat Top Beam",
        "Hermite Gaussian Beam",
        "Laguerre Gaussian Beam",
    ],
)
build_navigation_from_dir(pages_dict, "pages/diffraction_caculator", icon= "🎯")
build_navigation_from_dir(
    pages_dict,
    "pages/pde_caculator",
    icon="📐",
    page_order=["Sturm Liouville"],
)

pg = st.navigation(pages_dict)
pg.run()