# app.py
import streamlit as st
from pathlib import Path
import re

def build_navigation_from_dir(pages_dict = {}, base_dir="pages", icon = "📄"):
    base_path = Path(base_dir)
    py_files = sorted(list(base_path.rglob("*.py")))

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

    return pages_dict

pages_dict = {}
pages_dict[""] = [st.Page("pages/main.py", title="home", icon="🏠")]
build_navigation_from_dir(pages_dict, "pages/TMM", icon= "📊")
build_navigation_from_dir(pages_dict, "pages/Gaussain beams", icon= "🎯")
build_navigation_from_dir(pages_dict, "pages/Diffraction", icon= "🎯")

pg = st.navigation(pages_dict)
pg.run()