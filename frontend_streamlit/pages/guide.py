import streamlit as st

from ui.components import (
    render_brand_header,
    render_footer,
    render_info_card,
    render_page_heading,
    render_section_heading,
)


render_brand_header()
render_page_heading(
    "PANDUAN",
    "Satu langkah pada satu waktu.",
    "Panduan singkat untuk membaca tren, mencoba simulasi, dan memahami "
    "konteks estimasi.",
)

render_section_heading("CARA MENGGUNAKAN", "Mulai dengan tiga langkah sederhana.")
steps = st.columns(3)
with steps[0]:
    render_info_card(
        "LANGKAH 01",
        "Jelajahi data",
        "Buka Analisis Pasar untuk melihat ringkasan rata-rata historis berdasarkan genre dan platform.",
        "01",
    )
with steps[1]:
    render_info_card(
        "LANGKAH 02",
        "Atur skenario",
        "Di halaman Prediksi, pilih platform, genre, tahun, dan publisher yang tersedia.",
        "02",
    )
with steps[2]:
    render_info_card(
        "LANGKAH 03",
        "Baca hasil",
        "Lihat estimasi dan kategori; riwayat hasil yang tersimpan dapat dibuka kembali.",
        "03",
    )

render_section_heading("MEMBACA DENGAN KONTEKS", "Apa arti hasilnya?")
left, right = st.columns(2)
with left:
    st.markdown(
        '<div class="guide-card"><div class="eyebrow">KATEGORI</div>'
        "<h3>Niche · Standard · Blockbuster</h3>"
        "<p>Niche: di bawah 1 juta unit. Standard: 1 hingga kurang dari 5 juta unit. "
        "Blockbuster: 5 juta unit atau lebih.</p></div>",
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        '<div class="guide-card"><div class="eyebrow">BATASAN</div>'
        "<h3>Estimasi, bukan janji</h3>"
        "<p>Model menggunakan pola data historis 1980–2016 dengan empat fitur "
        "rilis. Nilai R² baseline rendah; hasil tidak menjamin penjualan aktual "
        "dan tidak menangkap seluruh faktor pasar.</p></div>",
        unsafe_allow_html=True,
    )

with st.expander("Variabel model"):
    st.markdown(
        "**Masukan:** Platform, Genre, Year, Publisher.  \n"
        "**Target:** Global_Sales dalam juta unit terjual.  \n"
        "Nama game bersifat opsional dan bukan fitur model. Penjualan regional "
        "juga tidak digunakan."
    )

render_footer()
