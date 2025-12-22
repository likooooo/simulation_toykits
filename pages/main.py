# app.py
import streamlit as st
from pathlib import Path
import re

# @st.cache_data
def show_markdown_file(file_path):
    path = Path(file_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        st.markdown(content, unsafe_allow_html=True)
    else:
        st.error(f"找不到文件: {file_path}")


st.set_page_config(page_title="Simulation toykits", layout="wide")

show_markdown_file("./assets/markdown/ui/wellcome.md")
show_markdown_file("./README.MD")
