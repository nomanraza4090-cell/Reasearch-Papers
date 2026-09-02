"""
literature_review.py
=====================
Organize a set of retrieved papers into a themed literature review with
inline references. This module performs the retrieval-adjacent
organization work; if an LLM is configured (see src/rag.py), the RAG layer
can use this structure as grounded context for narrative generation.
Without an LLM configured, generate_literature_review still returns a
complete, well-organized, purely extractive review (no fabrication).
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Optional

from src import config, research_gap

logger = config.get_logger(__name__)


def _theme_papers(papers: List[dict]) -> dict:
    """Group papers by their most prominent shared keyword/topic (heuristic)."""
    themes = defaultdict(list)
    frequent = research_gap._top_topics(papers, top_n=6)
    theme_words = [t["topic"] for t in frequent] or ["general"]

    for p in papers:
        text = f"{p.get('title','')} {p.get('fields_of_study','')}".lower()
        matched = next((w for w in theme_words if w in text), None)
        themes[matched or "other"].append(p)
    return dict(themes)


def generate_literature_review(topic: str, papers: List[dict]) -> dict:
    """
    Returns a structured literature review:
        {
          "topic": ...,
          "num_papers": ...,
          "themes": [ {"theme": ..., "papers": [...], "summary": ...}, ... ],
          "references": [ "Author (Year). Title. Venue.", ... ],
        }
    """
    if not papers:
        return {
            "topic": topic,
            "num_papers": 0,
            "themes": [],
            "references": [],
            "note": f"No papers were found for topic '{topic}'. Try a broader search query.",
        }

    themed = _theme_papers(papers)
    theme_sections = []
    for theme, theme_papers in sorted(themed.items(), key=lambda kv: -len(kv[1])):
        theme_papers_sorted = sorted(theme_papers, key=lambda p: p.get("citation_count", 0) or 0, reverse=True)
        summary_lines = []
        for p in theme_papers_sorted[:5]:
            year = p.get("year", "n.d.")
            summary_lines.append(f"{p.get('title','Untitled')} ({year}) examines this area; " + (p.get("tldr") or (p.get("abstract","")[:180] + "...")))
        theme_sections.append({
            "theme": theme,
            "num_papers": len(theme_papers),
            "papers": [
                {"title": p.get("title"), "year": p.get("year"), "authors": p.get("authors"), "url": p.get("url")}
                for p in theme_papers_sorted
            ],
            "summary": " ".join(summary_lines),
        })

    references = []
    for p in sorted(papers, key=lambda p: (p.get("year") or 0), reverse=True):
        author = (p.get("authors") or "Unknown").split(",")[0].strip()
        year = p.get("year", "n.d.")
        title = p.get("title", "Untitled")
        venue = p.get("venue", "")
        references.append(f"{author} et al. ({year}). {title}. {venue}.".strip())

    return {
        "topic": topic,
        "num_papers": len(papers),
        "themes": theme_sections,
        "references": references,
    }
