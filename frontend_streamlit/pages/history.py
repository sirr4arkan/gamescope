import pandas as pd
import streamlit as st

from ui.api_client import ApiError, get_history
from ui.components import (
    render_brand_header,
    render_category,
    render_error_card,
    render_footer,
    render_page_heading,
)


def _category_for(sales):
    if sales < 1:
        return "Niche"
    if sales < 5:
        return "Standard"
    return "Blockbuster"


render_brand_header()
render_page_heading(
    "RIWAYAT",
    "Jejak skenario yang tersimpan.",
    "Riwayat diambil dari penyimpanan lokal sistem. Data tetap tersedia saat "
    "aplikasi dijalankan kembali.",
)

try:
    history = get_history()
except ApiError:
    render_error_card("Riwayat prediksi belum dapat dimuat.")
    render_footer()
    st.stop()

if not history:
    st.markdown(
        '<div class="empty-card"><div class="eyebrow">BELUM ADA CATATAN</div>'
        "<h3>Riwayat prediksi akan muncul di sini.</h3>"
        "<p>Buat simulasi pada halaman Prediksi untuk menyimpan catatan pertama.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
else:
    records = []
    for row in history[:20]:
        try:
            sales = float(row["predicted_global_sales"])
            created_at = pd.to_datetime(row["created_at"], utc=True)
            records.append(
                {
                    "Waktu (UTC)": created_at.strftime("%Y-%m-%d %H:%M"),
                    "Nama game": row.get("game_title") or "—",
                    "Platform": row["platform"],
                    "Genre": row["genre"],
                    "Publisher": row["publisher"],
                    "Tahun": row["release_year"],
                    "Estimasi (juta unit terjual)": round(sales, 3),
                    "Kategori": _category_for(sales),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

    if not records:
        st.info("Belum ada catatan riwayat yang dapat ditampilkan.")
    else:
        st.markdown(
            '<div class="eyebrow">PREDIKSI TERBARU</div>'
            f'<h3 style="margin:.35rem 0 1rem;color:#304A37;'
            f'font-family:Georgia,serif">{len(records)} catatan</h3>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame(records),
            hide_index=True,
            width="stretch",
            height=min(500, 48 + 38 * len(records)),
        )
        selected_category = st.selectbox(
            "Lihat detail kategori",
            options=list(dict.fromkeys(row["Kategori"] for row in records)),
            key="history_category",
        )
        render_category(selected_category)

render_footer()
