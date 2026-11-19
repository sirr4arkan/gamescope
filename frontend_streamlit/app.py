from pathlib import Path

import streamlit as st

from ui.theme import apply_theme


ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title="GameScope — Analisis Penjualan Game",
    page_icon="G",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PAGES = [
    st.Page(ROOT / "pages" / "home.py", title="Beranda", icon=":material/home:", default=True),
    st.Page(ROOT / "pages" / "prediction.py", title="Prediksi", icon=":material/auto_awesome:"),
    st.Page(ROOT / "pages" / "analysis.py", title="Analisis Pasar", icon=":material/monitoring:"),
    st.Page(ROOT / "pages" / "model.py", title="Model AI", icon=":material/neurology:"),
    st.Page(ROOT / "pages" / "history.py", title="Riwayat", icon=":material/history:"),
    st.Page(ROOT / "pages" / "guide.py", title="Panduan", icon=":material/menu_book:"),
]

apply_theme()
current_page = st.navigation(PAGES, position="top")
current_page.run()
