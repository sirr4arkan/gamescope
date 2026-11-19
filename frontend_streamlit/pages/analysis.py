import streamlit as st

from ui.api_client import ApiError, explain_market_analysis, get_metadata
from ui.charts import render_sales_chart
from ui.components import (
    render_brand_header,
    render_error_card,
    render_footer,
    render_metric_card,
    render_page_heading,
    render_section_heading,
)


def _render_analysis_explanation(analysis_type, button_label):
    explanation_key = f"{analysis_type}_explanation"
    error_key = f"{analysis_type}_explanation_error"

    if st.button(
        button_label,
        key=f"{analysis_type}_explain",
        type="primary",
    ):
        with st.spinner("AI sedang menganalisis data grafik..."):
            try:
                explanation = explain_market_analysis(analysis_type)
            except ApiError as error:
                st.session_state[error_key] = str(error)
            else:
                st.session_state[explanation_key] = explanation
                st.session_state.pop(error_key, None)

    error_message = st.session_state.get(error_key)
    if error_message:
        st.error(error_message)

    explanation = st.session_state.get(explanation_key)
    if explanation:
        with st.container(border=True, key=f"{analysis_type}_explanation_card"):
            st.markdown(explanation)


render_brand_header()
render_page_heading(
    "ANALISIS PASAR",
    "Pola lama, perspektif baru.",
    "Rangkuman agregat Global Sales pada dataset bersih. Grafik menunjukkan "
    "rata-rata historis, bukan prediksi untuk judul tertentu.",
)

try:
    metadata = get_metadata()
except ApiError as error:
    render_error_card(f"Ringkasan pasar belum dapat dimuat: {error}")
    render_footer()
    st.stop()

dataset = metadata["dataset_summary"]
year_range = metadata["year_range"]
overview = st.columns(4)
with overview[0]:
    render_metric_card("Game bersih", f'{dataset["cleaned_rows"]:,}', "baris data")
with overview[1]:
    render_metric_card(
        "Rata-rata Penjualan Global",
        f'{dataset["mean_global_sales"]:.2f}',
        "juta unit terjual per game",
    )
with overview[2]:
    render_metric_card(
        "Median Penjualan Global",
        f'{dataset["median_global_sales"]:.2f}',
        "juta unit terjual per game",
    )
with overview[3]:
    render_metric_card(
        "Rentang data",
        f"{year_range[0]}–{year_range[1]}",
        "tahun rilis",
    )

render_section_heading(
    "RATA-RATA HISTORIS",
    "Penjualan rata-rata menurut kategori.",
    "Rata-rata mudah dipengaruhi sejumlah game dengan penjualan sangat tinggi.",
)
genre_column, platform_column = st.columns(2)
with genre_column:
    with st.container(border=True):
        st.markdown(
            '<div class="chart-title">Berdasarkan Genre</div>'
            '<div class="chart-description">Rata-rata Global Sales per genre, dalam juta unit terjual.</div>',
            unsafe_allow_html=True,
        )
        genre_values = metadata["average_sales_by_genre"]
        if genre_values:
            render_sales_chart(genre_values, "Genre", height=390)
        else:
            st.info("Belum ada data genre untuk ditampilkan.")
        _render_analysis_explanation(
            "genre_analysis",
            "Jelaskan Grafik Genre dengan AI",
        )
with platform_column:
    with st.container(border=True):
        st.markdown(
            '<div class="chart-title">Berdasarkan Platform</div>'
            '<div class="chart-description">Rata-rata Global Sales per platform, dalam juta unit terjual.</div>',
            unsafe_allow_html=True,
        )
        platform_values = metadata["average_sales_by_platform"]
        if platform_values:
            render_sales_chart(platform_values, "Platform", height=620)
        else:
            st.info("Belum ada data platform untuk ditampilkan.")
        _render_analysis_explanation(
            "platform_analysis",
            "Jelaskan Grafik Platform dengan AI",
        )

st.markdown(
    '<div class="notice-strip" style="margin-top:1.5rem">Dataset mencakup '
    f'{dataset["cleaned_rows"]:,} game rilis {year_range[0]}–{year_range[1]}. '
    "Agregat tidak mengendalikan tahun rilis atau perbedaan antara satu publisher "
    "dan lainnya; baca sebagai ringkasan deskriptif.</div>",
    unsafe_allow_html=True,
)
render_footer()
