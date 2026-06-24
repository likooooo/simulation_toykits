import streamlit as st
from pathlib import Path
import re

if "_sim_db_ready" not in st.session_state: