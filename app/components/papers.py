"""
app/components/papers.py
=========================
Responsible for rendering individual paper cards and paper detail views.
Pure Streamlit rendering logic -- no business logic lives here.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st


def _fmt_authors(authors: str, max_len: int = 90) -> str:
    if not authors:
        return "Unknown authors"
    return authors if len(authors) <= max_len else authors[:max_len].rstrip(", ") + ", et al."


def render_paper_card(paper: dict, show_score: bool = False, score_key: str = "similarity_score") -> None:
    """Render one paper as a clean, professional card."""
    title = paper.get("title", "Untitled")
    authors = _fmt_authors(paper.get("authors", ""))
    year = paper.get("year", "n.d.")
    venue = paper.get("venue", "")
    abstract = paper.get("abstract", "") or ""
    url = paper.get("url", "")
    citation_count = paper.get("citation_count", 0)
    score = paper.get(score_key)

    abstract_preview = abstract[:420] + ("..." if len(abstract) > 420 else "")

    badges = f'<span class="rm-badge">{year}</span>'
    if venue:
        badges += f'<span class="rm-badge">{venue}</span>'
    if citation_count is not None:
        badges += f'<span class="rm-badge rm-badge-accent">{citation_count} citations</span>'
    if show_score and score is not None:
        badges += f'<span class="rm-badge rm-badge-accent">score: {score:.3f}</span>'

    link_html = f'<a class="rm-link" href="{url}" target="_blank">View original paper →</a>' if url else ""

    st.markdown(
        f"""
        <div class="rm-card">
            <div class="rm-card-title">{title}</div>
            <div class="rm-card-meta">{authors}</div>
            <div style="margin-bottom:0.5rem;">{badges}</div>
            <div class="rm-abstract">{abstract_preview}</div>
            <div style="margin-top:0.5rem;">{link_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_paper_selector(papers: list, label: str, key: str) -> Optional[dict]:
    """Dropdown selector returning the chosen paper dict (used by comparison UI)."""
    if not papers:
        st.info("No papers available to select. Run a search first.")
        return None
    options = {f"{p.get('title', 'Untitled')} ({p.get('year', 'n.d.')})": p for p in papers}
    choice = st.selectbox(label, list(options.keys()), key=key)
    return options.get(choice)
