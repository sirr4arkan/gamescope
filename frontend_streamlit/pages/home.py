import base64
from pathlib import Path

import streamlit as st

from ui.api_client import ApiError, get_metadata
from ui.components import (
    render_footer,
    render_info_card,
    render_metric_card,
    render_section_heading,
)


ROOT = Path(__file__).resolve().parents[1]

st.markdown(
    '<div class="brand-lockup"><span class="brand-mark">G</span>'
    '<span class="brand-name">GameScope</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="brand-context">WAWASAN PASAR GAME</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="header-rule"></div>', unsafe_allow_html=True)

try:
    metadata = get_metadata()
except ApiError as error:
    metadata = None
    metadata_error = str(error)
else:
    metadata_error = ""

with st.container(key="home_hero"):
    hero_copy, hero_art = st.columns([1.15, 0.85], vertical_alignment="center")
    with hero_copy:
        st.markdown(
            '<div class="hero-copy"><span class="hero-kicker">SINYAL PASAR HISTORIS'
            "</span><h1>Kenali pasar.<br>Uji skenario berikutnya.</h1>"
            "<p>Jelajahi pola penjualan game lintas genre dan platform, lalu buat "
            "estimasi awal berdasarkan data historis.</p></div>",
            unsafe_allow_html=True,
        )
        if st.button("Coba simulasi prediksi  →", type="primary", key="home_predict"):
            st.switch_page("pages/prediction.py")
    with hero_art:
        illustration = ROOT / "assets" / "hero-forest.svg"
        encoded = base64.b64encode(illustration.read_bytes()).decode("ascii")
        st.markdown(
            f'<img class="hero-art" alt="Ilustrasi game controller di lanskap hutan" '
            f'src="data:image/svg+xml;base64,{encoded}">',
            unsafe_allow_html=True,
        )

if metadata is None:
    st.warning(
        "Ringkasan dataset/model belum dapat dimuat: " + metadata_error
    )
else:
    dataset = metadata["dataset_summary"]
    model = metadata["model_summary"]
    rows = dataset.get("cleaned_rows", 0)
    year_range = metadata["year_range"]
    platform_count = len(metadata["platforms"])
    genre_count = len(metadata["genres"])
    st.markdown("<div style='height: 1.3rem'></div>", unsafe_allow_html=True)
    columns = st.columns(4)
    with columns[0]:
        render_metric_card("Game dianalisis", f"{rows:,}", "data historis bersih")
    with columns[1]:
        render_metric_card("Platform", platform_count, "tersedia pada dataset")
    with columns[2]:
        render_metric_card("Genre", genre_count, "kategori historis")
    with columns[3]:
        render_metric_card(
            "Rentang tahun", f"{year_range[0]}–{year_range[1]}", "tahun rilis"
        )

render_section_heading(
    "FITUR GAMESCOPE",
    "Jelajahi data dan susun estimasi.",
    "Tiga fitur utama membantu membaca pola pasar sebelum mencoba sebuah skenario.",
)
cards = st.columns(3)
with cards[0]:
    render_info_card(
        "01 · SIMULASI",
        "Prediksi Penjualan Game",
        "Susun skenario rilis berdasarkan platform, genre, tahun, dan publisher.",
        "↗",
    )
with cards[1]:
    render_info_card(
        "02 · EKSPLORASI",
        "Analisis Pasar Historis",
        "Bandingkan rata-rata Global Sales historis berdasarkan genre dan platform.",
        "⌁",
    )
with cards[2]:
    render_info_card(
        "03 · PEMAHAMAN",
        "Wawasan Machine Learning",
        "Pahami fitur, metrik, dan keterbatasan model baseline yang digunakan.",
        "◉",
    )

render_section_heading("ALUR PREDIKSI", "Dari karakteristik rilis menuju estimasi.")
st.markdown(
    '<div class="process-strip" aria-label="Alur prediksi: karakteristik rilis menuju estimasi penjualan">'
    '<div class="process-step"><span>01 · MASUKAN</span><strong>Karakteristik Rilis</strong></div>'
    '<span class="process-arrow" aria-hidden="true">→</span>'
    '<div class="process-step"><span>02 · MODEL</span><strong>Prediksi Machine Learning</strong></div>'
    '<span class="process-arrow" aria-hidden="true">→</span>'
    '<div class="process-step"><span>03 · HASIL</span><strong>Estimasi Penjualan</strong></div>'
    "</div>",
    unsafe_allow_html=True,
)

if metadata is not None:
    raw_metrics = metadata["model_summary"]["raw_metrics"]
    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="notice-strip">Model baseline: '
        f'{metadata["model_summary"]["selected_model"]} &nbsp;·&nbsp; '
        f'R² {raw_metrics["r2"]:.3f} &nbsp;·&nbsp; '
        f'RMSE {raw_metrics["rmse"]:.3f} juta unit terjual. '
        "Angka evaluasi menggambarkan keterbatasan baseline, bukan akurasi tinggi."
        "</div>",
        unsafe_allow_html=True,
    )
render_footer()
