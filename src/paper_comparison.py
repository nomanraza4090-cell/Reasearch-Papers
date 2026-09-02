"""
paper_comparison.py
====================
Structured, side-by-side comparison of two (or more) papers across:
objective, methodology, dataset, results, strengths, limitations, and
research direction. Purely extractive from available metadata/abstract/tldr
fields -- never invents details a paper's record doesn't contain.
"""

from __future__ import annotations

import re
from typing import List

from src import config

logger = config.get_logger(__name__)

_METHOD_KEYWORDS = [
    "transformer", "cnn", "convolutional", "lstm", "bert", "gpt", "random forest",
    "svm", "regression", "clustering", "reinforcement learning", "neural network",
    "deep learning", "machine learning", "survey", "systematic review", "meta-analysis",
    "qualitative", "quantitative", "case study", "randomized", "cross-sectional",
]
_DATASET_KEYWORDS = [
    "imagenet", "mimic", "coco", "glue", "squad", "reddit", "twitter", "electronic health record",
    "ehr", "benchmark dataset", "public dataset", "corpus",
]
_LIMITATION_RE = re.compile(r"(limitation[s]?|however,|small sample|future work)[^.]*\.", re.IGNORECASE)
_RESULT_RE = re.compile(r"(result[s]?|we (?:find|found|show|demonstrate|achieve)|accuracy|outperform)[^.]*\.", re.IGNORECASE)


def _extract_field(text: str, keywords: List[str]) -> str:
    text_lower = (text or "").lower()
    found = sorted({k for k in keywords if k in text_lower})
    return ", ".join(found) if found else "Not explicitly stated in available metadata."


def _extract_snippet(text: str, pattern: re.Pattern, default: str) -> str:
    match = pattern.search(text or "")
    return match.group(0).strip() if match else default


def _summarize_paper(paper: dict) -> dict:
    abstract = paper.get("abstract", "") or ""
    return {
        "title": paper.get("title", "Untitled"),
        "objective": paper.get("tldr") or (abstract[:200] + ("..." if len(abstract) > 200 else "")),
        "methodology": _extract_field(f"{paper.get('title','')} {abstract}", _METHOD_KEYWORDS),
        "dataset": _extract_field(abstract, _DATASET_KEYWORDS),
        "results": _extract_snippet(abstract, _RESULT_RE, "Not explicitly stated in the abstract."),
        "strengths": "High citation impact" if (paper.get("citation_count") or 0) > 50 else "Not derivable from metadata alone.",
        "limitations": _extract_snippet(abstract, _LIMITATION_RE, "Not explicitly stated in the abstract."),
        "future_work": "See paper for future-work discussion." if "future work" in abstract.lower() else "Not explicitly stated.",
        "year": paper.get("year"),
        "venue": paper.get("venue"),
        "citation_count": paper.get("citation_count", 0),
        "url": paper.get("url"),
    }


def compare_papers(paper_a: dict, paper_b: dict) -> dict:
    """Return a structured comparison dict between two papers."""
    if not paper_a or not paper_b:
        raise ValueError("compare_papers requires two non-empty paper records.")
    return {
        "paper_a": _summarize_paper(paper_a),
        "paper_b": _summarize_paper(paper_b),
    }


def compare_multiple(papers: List[dict]) -> dict:
    """Extend comparison to N papers (used by the Streamlit multi-select UI)."""
    if len(papers) < 2:
        raise ValueError("compare_multiple requires at least two papers.")
    return {f"paper_{chr(97+i)}": _summarize_paper(p) for i, p in enumerate(papers)}
