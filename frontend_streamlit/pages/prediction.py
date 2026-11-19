import streamlit as st

from ui.api_client import ApiError, get_metadata, predict
from ui.components import (
    render_brand_header,
    render_category,
    render_error_card,
    render_footer,
    render_page_heading,
)


render_brand_header()
render_page_heading(
    "PREDIKSI",
    "Coba satu skenario rilis.",
    "Atur karakteristik rilis untuk melihat estimasi Global Sales "
    "berdasarkan pola historis dataset.",
)

try:
    metadata = get_metadata()
except ApiError:
    render_error_card(
        "Pilihan formulir belum dapat dimuat dari aplikasi lokal."
    )
    render_footer()
    st.stop()

year_min, year_max = (int(value) for value in metadata["year_range"])
with st.container(border=True):
    st.markdown('<div class="eyebrow">SKENARIO RILIS GAME</div>'
                '<h3 style="margin:.35rem 0 1.1rem;color:#304A37;'
                'font-family:Georgia,serif">Ceritakan konsep game-mu</h3>',
                unsafe_allow_html=True)
    with st.form("prediction_form"):
        title = st.text_input("Nama game (opsional)", max_chars=200)
        first_row = st.columns(2)
        with first_row[0]:
            platform = st.selectbox("Platform", metadata["platforms"])
        with first_row[1]:
            genre = st.selectbox("Genre", metadata["genres"])
        second_row = st.columns(2)
        with second_row[0]:
            release_year = st.number_input(
                "Tahun rilis",
                min_value=year_min,
                max_value=year_max,
                value=year_max,
                step=1,
            )
        with second_row[1]:
            publisher = st.selectbox("Publisher", metadata["publishers"])
        submitted = st.form_submit_button(
            "Lihat estimasi  →", type="primary", use_container_width=True
        )

if submitted:
    payload = {
        "game_title": title.strip(),
        "platform": platform,
        "genre": genre,
        "release_year": int(release_year),
        "publisher": publisher,
    }
    try:
        with st.spinner("Menganalisis skenario..."):
            result = predict(payload)
        st.session_state["latest_prediction"] = result
    except ApiError as error:
        st.error(str(error))

result = st.session_state.get("latest_prediction")
if result:
    st.markdown("<div style='height: 1.4rem'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="result-card"><div class="eyebrow">ESTIMASI PENJUALAN GLOBAL</div>'
        f'<div class="result-value">{result["predicted_global_sales"]:.3f}'
        '<span style="font-size:.38em;letter-spacing:0"> juta unit terjual</span></div>'
        '<div class="result-note">Kategori hasil: '
        f'{result["category"]}. Estimasi ini mencerminkan pola historis dataset, '
        "bukan jaminan penjualan aktual.</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: .7rem'></div>", unsafe_allow_html=True)
    render_category(result["category"])

st.markdown(
    '<div class="notice-strip" style="margin-top:1.6rem">Rentang tahun mengikuti '
    f'dataset historis ({year_min}–{year_max}). Prediksi merupakan estimasi awal, '
    "bukan jaminan penjualan aktual.</div>",
    unsafe_allow_html=True,
)
render_footer()
