import html

import streamlit as st


def render_brand_header():
    brand, context = st.columns([1.2, 4], vertical_alignment="center")
    with brand:
        st.markdown(
            '<div class="brand-lockup"><span class="brand-mark">G</span>'
            '<span class="brand-name">GameScope</span></div>',
            unsafe_allow_html=True,
        )
    with context:
        st.markdown(
            '<div class="brand-context">WAWASAN PASAR GAME</div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="header-rule"></div>', unsafe_allow_html=True)


def render_footer():
    st.markdown(
        '<div class="site-footer"><span>GAMESCOPE</span>'
        "<span>Estimasi historis, bukan jaminan penjualan aktual.</span></div>",
        unsafe_allow_html=True,
    )


def render_page_heading(eyebrow, title, description):
    safe_eyebrow = html.escape(eyebrow)
    safe_title = html.escape(title)
    safe_description = html.escape(description)
    st.markdown(
        f'<div class="page-heading"><div class="eyebrow">{safe_eyebrow}</div>'
        f'<h1>{safe_title}</h1><p>{safe_description}</p></div>',
        unsafe_allow_html=True,
    )


def render_section_heading(eyebrow, title, description=None):
    body = (
        f'<p>{html.escape(description)}</p>'
        if description
        else ""
    )
    st.markdown(
        f'<div class="section-heading"><span class="eyebrow">'
        f'{html.escape(eyebrow)}</span><h2>{html.escape(title)}</h2>{body}</div>',
        unsafe_allow_html=True,
    )


def render_metric_card(label, value, note=""):
    st.markdown(
        '<div class="metric-card"><div class="metric-label">'
        f"{html.escape(str(label))}</div><div class=\"metric-value\">"
        f"{html.escape(str(value))}</div><div class=\"metric-note\">"
        f"{html.escape(str(note))}</div></div>",
        unsafe_allow_html=True,
    )


def render_info_card(kicker, title, description, icon=""):
    st.markdown(
        '<div class="info-card">'
        f'<div class="info-icon">{html.escape(icon)}</div>'
        f'<div class="eyebrow">{html.escape(kicker)}</div>'
        f"<h3>{html.escape(title)}</h3>"
        f"<p>{html.escape(description)}</p></div>",
        unsafe_allow_html=True,
    )


def render_error_card(message):
    st.markdown(
        '<div class="empty-card"><div class="eyebrow">BELUM TERSEDIA</div>'
        f"<h3>{html.escape(message)}</h3>"
        "<p>Coba kembali beberapa saat lagi.</p></div>",
        unsafe_allow_html=True,
    )


def render_category(category):
    safe_category = html.escape(category)
    category_class = {
        "Niche": "category-niche",
        "Standard": "category-standard",
        "Blockbuster": "category-blockbuster",
    }.get(category, "category-standard")
    st.markdown(
        f'<span class="category-pill {category_class}">{safe_category}</span>',
        unsafe_allow_html=True,
    )
