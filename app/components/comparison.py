"""
app/components/comparison.py
=============================
Streamlit UI for comparing two or more papers side by side. All rendering
logic lives here; the actual comparison is computed by
src/paper_comparison.py.
"""

from __future__ import annotations

import streamlit as st

from src import paper_comparison
from app.components.papers import render_paper_selector

_FIELDS = [
    ("objective", "Objective"),
    ("methodology", "Methodology"),
    ("dataset", "Dataset"),
    ("results", "Results"),
    ("strengths", "Strengths"),
    ("limitations", "Limitations"),
    ("future_work", "Future Work"),
]


def render_comparison_page():
    st.subheader("Paper Comparison")
    st.caption("Select two or more papers from your last search to compare them side by side.")

    papers = st.session_state.get("last_search_results", [])
    if not papers or len(papers) < 2:
        st.info("Run a search with at least two results first (see the Semantic Search page).")
        return

    num_papers = st.radio("Number of papers to compare", [2, 3], horizontal=True)

    selected = []
    cols = st.columns(num_papers)
    for i, col in enumerate(cols):
        with col:
            paper = render_paper_selector(papers, label=f"Paper {chr(65 + i)}", key=f"compare_select_{i}")
            selected.append(paper)

    if any(p is None for p in selected):
        return

    if st.button("Compare", type="primary"):
        try:
            comparison = paper_comparison.compare_multiple(selected)
        except ValueError as e:
            st.error(str(e))
            return

        keys = list(comparison.keys())
        table_cols = st.columns(len(keys))
        for col, key in zip(table_cols, keys):
            with col:
                st.markdown(f"**{comparison[key]['title']}**")
                st.caption(f"{comparison[key].get('venue','')} ({comparison[key].get('year','n.d.')})")

        st.markdown("---")
        for field_key, field_label in _FIELDS:
            st.markdown(f"**{field_label}**")
            row_cols = st.columns(len(keys))
            for col, key in zip(row_cols, keys):
                with col:
                    st.write(comparison[key].get(field_key, "—"))
            st.markdown("")
