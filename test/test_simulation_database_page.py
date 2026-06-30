"""Tests for simulation_database Streamlit page helpers."""

from __future__ import annotations


def test_bootstrap_root_tree_expansion(sim_db, mock_streamlit_session) -> None:
    from simulation_database.database_precompiling import load_or_build_database_index
    from simulation_database.database_ui import build_tree_nodes_for_panel, get_tree_children
    from simulation_database.page import ROOT_BOOTSTRAP_DONE_KEY, _bootstrap_root_tree_expansion
    from simulation_database.workspace import ensure_sim_workspace_ui

    load_or_build_database_index(sim_db)
    ui = ensure_sim_workspace_ui()
    assert not ui.expanded_paths

    _bootstrap_root_tree_expansion(sim_db, ui)

    root_children = get_tree_children(sim_db, [], ui.children_cache)
    expected = {m["path_id"] for m in root_children if not m["is_leaf"]}
    assert ui.expanded_paths == expected
    assert mock_streamlit_session.get(ROOT_BOOTSTRAP_DONE_KEY)

    nodes = build_tree_nodes_for_panel(sim_db, ui.expanded_paths, ui.children_cache)
    for node in nodes:
        if not node["is_leaf"] and node["path_id"] in expected:
            assert node["children"], f"{node['path_id']} should have loaded children"
