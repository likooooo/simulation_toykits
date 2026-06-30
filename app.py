import streamlit as st
from pathlib import Path
import re
import simulation

# Streamlit 会在交互/切页/热重载时反复 rerun 脚本；这里保证每个会话只打印一次。
if not st.session_state.get("_simulation_banner_printed", False):
    simulation.print_simulation_banner()
    st.session_state["_simulation_banner_printed"] = True

from simulation_database.workspace import WORKSPACE_SCHEMA

if st.session_state.get("_sim_db_ready_schema") != WORKSPACE_SCHEMA:
    import simulation_database_parser as sdp

    from simulation_database.workspace import ensure_workspace_initialized
    from common import (
        build_filmstack_preset_catalog,
        get_default_material_path_keys,
        get_filmstack_template_by_id,
        get_required_default_material_names,
    )
    from toykits_config import (
        DEFAULT_SPECTRUM_PATH,
        resolve_filmstack_initial_defaults,
    )

    catalog = build_filmstack_preset_catalog()
    template_map = get_filmstack_template_by_id()
    st.session_state["_filmstack_preset_catalog"] = catalog
    st.session_state["_filmstack_template_by_id"] = template_map
    prep_log: list[str] = []
    sim_db = None
    with st.spinner("正在准备仿真数据库..."):
        try:
            sim_db = sdp.get_simulation_database(init=True)
            prep_log.append(f"database root: {sim_db.root_path()}")
            prep_log.append(f"local path: {sim_db.local_path()}")
            from simulation_database.database_precompiling import load_or_build_database_index

            index = load_or_build_database_index(sim_db)
            prep_log.append(f"precompiled index: {index.leaf_count} entries")
            ensure_workspace_initialized(
                sim_db,
                material_path_keys=get_default_material_path_keys(),
                spectrum_path_keys=[DEFAULT_SPECTRUM_PATH],
                strict=True,
                required_material_names=get_required_default_material_names(),
            )
            prep_log.append("ready")
            st.session_state["_sim_db_ready_schema"] = WORKSPACE_SCHEMA
        except Exception as exc:
            prep_log.append(f"prepare failed: {exc}")
            st.error(
                "仿真数据库或默认材料加载失败，Filmstack 预设可能无法使用。\n"
                f"详情: {exc}\n"
                "请确认已运行 `python scripts/build_toykits.py` 并重新启动 app。"
            )
    if st.session_state.get("_sim_db_ready_schema") == WORKSPACE_SCHEMA:
        st.session_state["_filmstack_initial_defaults"] = resolve_filmstack_initial_defaults(
            catalog.valid_preset_ids,
            template_by_id=template_map,
        )
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


def build_navigation_from_dir(
    pages_dict=None,
    base_dir="pages",
    icon="📄",
    page_order=None,
    title_overrides=None,
):
    if pages_dict is None:
        pages_dict = {}
    base_path = Path(base_dir)
    py_files = sorted(list(base_path.rglob("*.py")))
    group_name = None
    overrides = title_overrides or {}

    for py_file in py_files:
        relative_path = py_file.relative_to(base_path.parent)
        parts = relative_path.parts
        group_name = parts[0] if len(parts) > 1 else "Others"
        display_name = overrides.get(py_file.stem)
        if display_name is None:
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
pages_dict[""] = [st.Page("pages/main.py", title="Home", icon="🏠", default=True)]
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
    title_overrides={
        "freehand optimization": "Freehand Optimization",
    },
)
build_navigation_from_dir(
    pages_dict,
    "pages/gaussian_optics_toolkits",
    icon="🎯",
    page_order=[
        "Plane Wave",
        "Quadratic Wave",
        "Spherical Wave",
        "Flat-Top Beam",
        "Hermite-Gaussian Beam",
        "Laguerre-Gaussian Beam",
    ],
    title_overrides={
        "flat top beam": "Flat-Top Beam",
        "hermite gaussian beam": "Hermite-Gaussian Beam",
        "laguerre gaussian beam": "Laguerre-Gaussian Beam",
    },
)
build_navigation_from_dir(
    pages_dict,
    "pages/simulation_toykits",
    icon="📐",
    page_order=["Sturm-Liouville", "Time-Dependent Sturm-Liouville"],
    title_overrides={
        "sturm_liouville": "Sturm-Liouville",
        "time_dependent_sturm_liouville": "Time-Dependent Sturm-Liouville",
    },
)

from filmstack_simulation.help_texts import fs_compare_artifact_available

_hidden_pages = [
    st.Page(
        "pages/_hidden/filmstack_formula_usage.py",
        title="多层膜构建指令",
        url_path="filmstack-formula-usage",
        visibility="hidden",
    ),
]
if fs_compare_artifact_available():
    _hidden_pages.insert(
        0,
        st.Page(
            "pages/_hidden/fs_baseline_compare.py",
            title="Filmstack 基准测试",
            url_path="fs-baseline-compare",
            visibility="hidden",
        ),
    )
pages_dict["_hidden"] = _hidden_pages

pg = st.navigation(pages_dict)
from simulation_database.page import mark_streamlit_page_transition

mark_streamlit_page_transition(pg)
pg.run()
