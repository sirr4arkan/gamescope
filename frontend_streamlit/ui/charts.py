import altair as alt
import pandas as pd
import streamlit as st


WHITE = "#FFFFFF"
SAGE = "#C4D2B4"
SAGE_SOFT = "#E9EFE2"
OLIVE = "#96AA83"
FOREST = "#456B4D"
DEEP_FOREST = "#304A37"
PALETTE = ("forest", "olive", "sage")
COLORS = (FOREST, OLIVE, SAGE)


def render_sales_chart(values, category_name, height):
    rows = sorted(values.items(), key=lambda item: item[1])
    data = pd.DataFrame(
        [
            {
                category_name: str(name),
                "Sales": float(sales),
                "Tone": PALETTE[index % len(PALETTE)],
            }
            for index, (name, sales) in enumerate(rows)
        ]
    )
    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            x=alt.X(
                "Sales:Q",
                title="Rata-rata (juta unit terjual)",
                scale=alt.Scale(zero=True),
                axis=alt.Axis(grid=True, labelColor=DEEP_FOREST, titleColor=DEEP_FOREST),
            ),
            y=alt.Y(
                f"{category_name}:N",
                title=None,
                sort=alt.SortField(field="Sales", order="ascending"),
                axis=alt.Axis(labelColor=DEEP_FOREST, titleColor=DEEP_FOREST),
            ),
            color=alt.Color(
                "Tone:N",
                scale=alt.Scale(domain=list(PALETTE), range=list(COLORS)),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip(f"{category_name}:N", title=category_name),
                alt.Tooltip(
                    "Sales:Q",
                    title="Rata-rata (juta unit terjual)",
                    format=".4f",
                ),
            ],
        )
        .properties(height=height, background=WHITE)
        .configure(background=WHITE, font="Aptos")
        .configure_view(fill=WHITE, stroke=None)
        .configure_axis(
            domainColor=SAGE,
            gridColor=SAGE_SOFT,
            gridOpacity=0.85,
            tickColor=SAGE,
            labelColor=DEEP_FOREST,
            titleColor=DEEP_FOREST,
        )
    )
    st.altair_chart(
        chart,
        theme=None,
        width="stretch",
        key=f"sales-chart-{category_name.lower()}",
        alt=f"Rata-rata Global Sales berdasarkan {category_name}",
    )
