import streamlit as st
from pathlib import Path
import re

# 程序开始前把 docker_artifacts 加入 sys.path，core 内可直接 import simulation
from core import simulation_loader
simulation_loader.ensure_artifacts_on_path()

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
pages_dict[""] = [st.Page("pages/main.py", title="home", icon="🏠")]
build_navigation_from_dir(
    pages_dict,
    "pages/fresnel_caculator",
    icon="📊",
    page_order=["Material Database", "Films", "Spectral Curve", "Angular Spectral Map"],
)
build_navigation_from_dir(pages_dict, "pages/beams_caculator", icon= "🎯")
build_navigation_from_dir(pages_dict, "pages/diffraction_caculator", icon= "🎯")

pg = st.navigation(pages_dict)
pg.run()