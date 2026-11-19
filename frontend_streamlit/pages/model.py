import streamlit as st

from ui.api_client import ApiError, get_metadata
from ui.components import (
    render_brand_header,
    render_footer,
    render_metric_card,
    render_page_heading,
    render_section_heading,
)
from ui.diagrams import render_ml_workflow, render_system_architecture


render_brand_header()
render_page_heading(
    "MODEL AI",
    "Baseline yang transparan.",
    "Kenali data, fitur, dan metrik yang menjadi dasar setiap estimasi. "
    "Model sengaja dipertahankan sebagai baseline Linear Regression.",
)

try:
    metadata = get_metadata()
except ApiError as error:
    metadata = None
    metadata_error = str(error)
else:
    metadata_error = ""

if metadata is None:
    st.warning(
        "Ringkasan model belum dapat dimuat. " + metadata_error + " Diagram tetap tersedia."
    )
else:
    summary = metadata["model_summary"]
    raw = summary["raw_metrics"]
    clipped = summary["clipped_metrics"]

    metrics = st.columns(3)
    with metrics[0]:
        render_metric_card("R²", f'{raw["r2"]:.3f}', "evaluasi output mentah")
    with metrics[1]:
        render_metric_card(
            "RMSE", f'{raw["rmse"]:.3f}', "juta unit terjual · output mentah"
        )
    with metrics[2]:
        render_metric_card(
            "MAE", f'{raw["mae"]:.3f}', "juta unit terjual · output mentah"
        )

    st.markdown("<div style='height: .7rem'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="notice-strip">Metrik setelah pembatasan prediksi minimum nol: '
        f'R² {clipped["r2"]:.3f} · RMSE {clipped["rmse"]:.3f} · '
        f'MAE {clipped["mae"]:.3f} juta unit terjual. Evaluasi ini tidak menunjukkan '
        "akurasi tinggi; hasil merupakan baseline dengan keterbatasan yang nyata."
        "</div>",
        unsafe_allow_html=True,
    )

animate_diagrams = not st.session_state.get("model_diagrams_animation_played", False)
render_section_heading("ALUR MODEL", "Dari dataset menuju estimasi.")
render_ml_workflow(animate=animate_diagrams)

render_section_heading(
    "ARSITEKTUR SISTEM",
    "Hubungan antarkomponen aplikasi.",
    "Streamlit memuat metadata dan model dari artifacts/, menghitung prediksi "
    "secara lokal, lalu memakai Supabase dan Groq melalui HTTPS.",
)
render_system_architecture(animate=animate_diagrams)
st.session_state["model_diagrams_animation_played"] = True

if metadata is not None:
    dataset = metadata["dataset_summary"]
    render_section_heading("RINGKASAN MODEL", "Apa yang masuk ke model.")
    details = st.columns(2)
    with details[0]:
        st.markdown(
            '<div class="model-card"><div class="eyebrow">FITUR INPUT · X</div>'
            "<h3>Empat sinyal rilis</h3>"
            f'<p>Platform: {len(metadata["platforms"])} kategori · Genre: '
            f'{len(metadata["genres"])} kategori · Year: {metadata["year_range"][0]}–'
            f'{metadata["year_range"][1]} · Publisher: {len(metadata["publishers"])} '
            "kategori. Fitur kategorikal diproses dengan OneHotEncoder di dalam "
            "Pipeline.</p></div>",
            unsafe_allow_html=True,
        )
    with details[1]:
        st.markdown(
            '<div class="model-card"><div class="eyebrow">TARGET · y</div>'
            "<h3>Global Sales</h3>"
            f'<p>{dataset["cleaned_rows"]:,} baris bersih; nilai target median '
            f'{dataset["median_global_sales"]:.2f} juta unit terjual. Nama game hanya metadata, '
            "dan penjualan regional tidak digunakan sebagai fitur.</p></div>",
            unsafe_allow_html=True,
        )

    with st.expander("Cara membaca metrik"):
        st.markdown(
            "- **R²** mengukur seberapa jauh variasi target dijelaskan model; "
            "nilai yang rendah menunjukkan daya jelas terbatas.\n"
            "- **RMSE** memberi penalti lebih besar pada kesalahan besar dan "
            "dinyatakan dalam juta unit terjual.\n"
            "- **MAE** adalah rata-rata besarnya kesalahan absolut, juga dalam juta unit terjual.\n"
            "- Output API dibatasi minimum nol, sehingga metrik nilai mentah dan "
            "metrik setelah pembatasan dilaporkan terpisah."
        )

render_footer()
