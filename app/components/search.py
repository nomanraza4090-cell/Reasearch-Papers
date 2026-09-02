"""
app/components/search.py
=========================
Streamlit UI for the Semantic Search page. All Streamlit-specific rendering
logic lives here; business logic stays in src/semantic_search.py and
src/ranking.py.
"""

from __future__ import annotations

import streamlit as st

from src import semantic_search, ranking, config
from app.components.papers import render_paper_card


def render_search_page():
    st.subheader("Semantic Search")
    st.caption("Search is meaning-based, not simple keyword matching.")

    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("Search research papers...", key="search_query", label_visibility="collapsed",
                               placeholder="Search research papers...")
    with col2:
        top_k = st.selectbox("Results", [5, 10, 20], index=1, label_visibility="collapsed")

    apply_ranking = st.checkbox("Apply relevance ranking (semantic + citations + recency + quality)", value=True)

    if not query:
        st.info("Enter a query above to search the paper corpus semantically.")
        return

    try:
        results = semantic_search.search_papers(query, top_k=top_k)
    except FileNotFoundError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Search failed: {e}")
        return

    if not results:
        st.warning("No papers matched this query.")
        return

    if apply_ranking:
        results = ranking.compute_scores(results)

    st.session_state["last_search_results"] = results
    st.write(f"**{len(results)} paper(s) found** for *\"{query}\"*")

    for paper in results:
        render_paper_card(paper, show_score=True, score_key="final_score" if apply_ranking else "similarity_score")
