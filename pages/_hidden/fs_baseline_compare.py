"""Hidden viewer for fs baseline vs toykits compare HTML (artifact-only, no static sync)."""

from __future__ import annotations

import streamlit as st

from filmstack_simulation.help_texts import fs_compare_artifact_path

st.set_page_config(page_title="基准测试", layout="wide")

artifact = fs_compare_artifact_path()
if artifact is None or not artifact.is_file():
    st.stop()

st.iframe(str(artifact))
