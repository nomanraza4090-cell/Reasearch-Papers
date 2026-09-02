"""
app/components/analytics.py
============================
Streamlit UI for the Dashboard page: corpus-level metrics and charts.
Reads directly from the final dataset and the vector index -- no
recomputation of business logic here.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config, data_loader


def _metric_tile(label: str, value) -> str:
    return f"""
    <div class="rm-metric">
        <div class="rm-metric-value">{value}</div>
        <div class="rm-metric-label">{label}</div>
    </div>
    """


def render_dashboard_page():
    st.subheader("Dashboard")

    try:
        df = data_loader.load_final_dataset()
    except data_loader.DataLoadError as e:
        st.error(str(e))
        return

    num_indexed = 0
    index_status = "Not built"
    try:
        from src import vector_store as vs_mod
        store = vs_mod.VectorStore.load()
        num_indexed = len(store.metadata)
        index_status = f"Ready ({store.backend} backend)"
    except Exception:
        pass

    col_map = data_loader.get_available_columns(df)
    years = sorted(df[col_map["year"]].dropna().unique().tolist()) if col_map.get("year") else []
    avg_citations = df[col_map["citation_count"]].mean() if col_map.get("citation_count") else None
    num_fields = df[col_map["fields_of_study"]].nunique() if col_map.get("fields_of_study") else 0

    cols = st.columns(4)
    with cols[0]:
        st.markdown(_metric_tile("Total Papers", len(df)), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(_metric_tile("Indexed for Search", num_indexed), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(_metric_tile("Years Covered", f"{years[0]}–{years[-1]}" if years else "N/A"), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(_metric_tile("Avg. Citations", f"{avg_citations:.0f}" if avg_citations else "N/A"), unsafe_allow_html=True)

    st.markdown("<hr class='rm-section-divider'/>", unsafe_allow_html=True)
    st.caption(f"Search index status: **{index_status}**  ·  Fields of study represented: **{num_fields}**")

    left, right = st.columns(2)
    with left:
        if col_map.get("year"):
            st.markdown("**Papers by Year**")
            year_counts = df[col_map["year"]].dropna().astype(int).value_counts().sort_index()
            st.bar_chart(year_counts)
    with right:
        if col_map.get("fields_of_study"):
            st.markdown("**Top Fields of Study**")
            field_counts = df[col_map["fields_of_study"]].value_counts().head(10)
            st.bar_chart(field_counts)

    if col_map.get("venue"):
        st.markdown("**Top Venues**")
        st.dataframe(
            df[col_map["venue"]].value_counts().head(10).rename_axis("Venue").reset_index(name="Papers"),
            use_container_width=True, hide_index=True,
        )
