from pathlib import Path

import streamlit as st


THEME_PATH = Path(__file__).with_name("styles.css")


def apply_theme():
    css = THEME_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
