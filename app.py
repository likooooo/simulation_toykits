import streamlit as st
from pathlib import Path
import re
import simulation

if "_sim_db_ready" not in st.session_state:
    import simulation_database_parser as sdp

    from simulation_database.workspace import ensure_workspace_initialized
    from toykits_config import (
        DEFAULT_MATERIAL_PATH_KEYS,
        DEFAULT_SPECTRUM_PATH,
        FILMSTACK_PRESET_CATALOG,
        resolve_filmstack_initial_defaults,
    )

    prep_log: list[str] = []
    with st.spinner("正在准备仿真数据库..."):
        try:
            sim_db = sdp.get_simulation_database(init=True)
            prep_log.append(f"database root: {sim_db.root_path()}")
            prep_log.append(f"local path: {sim_db.local_path()}")
            prep_log.append("ready")
            ensure_workspace_initialized(
                sim_db,
                material_path_keys=DEFAULT_MATERIAL_PATH_KEYS,
                spectrum_path_keys=[DEFAULT_SPECTRUM_PATH],
            )
        except Exception as exc:
            sim_db = None
            prep_log.append(f"prepare failed: {exc}")
    st.session_state["_filmstack_initial_defaults"] = resolve_filmstack_initial_defaults(
        FILMSTACK_PRESET_CATALOG.valid_preset_ids
    )
    st.session_state["_sim_db_ready"] = True
    st.session_state["_sim_db_prepare_log"] = prep_log

st.markdown(
    """
    <style>
    /* Streamlit 列宽 / 滚动条抖动修复 */
    html {
        overflow-y: scroll !important;
    }

    [data-testid="stColumn"] {
        flex: 1 1 45% !important;
        min-width: 0 !important;
    }

    [data-testid="stHorizontalBlock"] {
        width: 100% !important;
        display: flex !important;
        flex-wrap: nowrap !important;
        align-items: flex-start !important;
    }

    [data-testid="stAppViewBlockContainer"] {
        padding-right: 2rem !important;
        padding-left: 2rem !important;
    }

    [data-testid="stAppViewContainer"] img {
        max-width: 100%;
        height: auto;
        object-fit: contain;
    }

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
    "pages/filmstack_toolkits",
    icon="📊",
    page_order=[
        "Simulation Database",
        "Filmstack Simulation",
        "Freehand Optimization",
        "Diffraction Angle",
    ],
)
build_navigation_from_dir(
    pages_dict,
    "pages/gaussian_optics_toolkits",
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
build_navigation_from_dir(
    pages_dict,
    "pages/simulation_toykits",
    icon="📐",
    page_order=["Sturm Liouville", "Time Dependent Sturm Liouville"],
)

pg = st.navigation(pages_dict)
pg.run()
