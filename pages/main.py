import streamlit as st

from common import show_markdown_file

st.set_page_config(page_title="Simulation-toykits", layout="wide")

show_markdown_file("./docs/welcome.md")
