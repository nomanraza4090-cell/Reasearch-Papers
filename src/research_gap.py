"""
research_gap.py
================
Analyze a set of retrieved papers and surface:
  - frequently studied topics
  - underrepresented topics
  - missing concept combinations
  - methodological / dataset limitations mentioned in abstracts
  - contradictory findings (heuristic keyword-based flagging)
  - future-work suggestions
  - potential (speculative) research directions

Every finding is tagged with an `evidence_type` of either "observed"
(directly derived from the supplied papers' text/metadata) or
"hypothesis" (a suggested, unverified research direction). Hypotheses are
never presented as established fact.
"""

from __future__ import annotations

import re
from collections import Counter
from itertools import combinations
from typing import List

from src import config

logger = config.get_logger(__name__)

_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "with",
    "is", "are", "was", "were", "this", "that", "these", "those", "we",
    "our", "study", "paper", "using", "based", "from", "by", "as", "at",
    "into", "such", "can", "be", "which", "it", "its", "their",
}

_LIMITATION_PATTERNS = [
    r"limitation[s]?", r"limited (?:by|to|in)", r"small(?:er)? (?:sample|dataset)",
    r"future work", r"future research", r"remains? (?:an? )?(?:open|challenge)",
    r"not (?:yet |been )?(?:fully )?(?:explored|addressed|studied)",
    r"lack(?:s|ing)? of", r"insufficient", r"further (?:study|research|investigation)",
]
_LIMITATION_RE = re.compile("|".join(_LIMITATION_PATTERNS), re.IGNORECASE)

_CONTRADICTION_PATTERNS = [
    r"in contrast", r"contrary to", r"however,? (?:prior|previous|earlier)",
    r"conflicting", r"inconsistent (?:with|results|findings)", r"disagree[s]?",
    r"unlike (?:prior|previous|earlier)",
]
_CONTRADICTION_RE = re.compile("|".join(_CONTRADICTION_PATTERNS), re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", (text or "").lower())
    return [w for w in words if w not in _STOPWORDS]


def _top_topics(papers: List[dict], top_n: int = 10) -> List[dict]:
    counter = Counter()
    for p in papers:
        text = f"{p.get('title', '')} {p.get('fields_of_study', '')} {p.get('research_topic', '')}"
        counter.update(set(_tokenize(text)))
    return [{"topic": t, "paper_count": c} for t, c in counter.most_common(top_n)]


def _underrepresented_topics(papers: List[dict], min_count: int = 1, max_count: int = 2) -> List[dict]:
    counter = Counter()
    for p in papers:
        text = f"{p.get('title', '')} {p.get('fields_of_study', '')}"
        counter.update(set(_tokenize(text)))
    return [
        {"topic": t, "paper_count": c}
        for t, c in counter.items()
        if min_count <= c <= max_count
    ][:15]


def _missing_combinations(top_topics: List[dict], papers: List[dict]) -> List[dict]:
    """Flag pairs of frequent topics that never co-occur in the same paper."""
    topic_words = [t["topic"] for t in top_topics[:8]]
    paper_topic_sets = [set(_tokenize(f"{p.get('title','')} {p.get('fields_of_study','')}")) for p in papers]

    missing = []
    for a, b in combinations(topic_words, 2):
        co_occurs = any(a in s and b in s for s in paper_topic_sets)
        if not co_occurs:
            missing.append({"combination": f"{a} + {b}", "evidence_type": "hypothesis"})
    return missing[:10]


def _find_pattern_mentions(papers: List[dict], pattern: re.Pattern, field: str = "abstract", limit: int = 10) -> List[dict]:
    mentions = []
    for p in papers:
        text = p.get(field, "") or ""
        for match in pattern.finditer(text):
            start = max(0, match.start() - 60)
            end = min(len(text), match.end() + 60)
            snippet = text[start:end].strip()
            mentions.append({
                "paper_id": p.get("paper_id", ""),
                "title": p.get("title", ""),
                "snippet": f"...{snippet}...",
                "evidence_type": "observed",
            })
            break  # one mention per paper is enough signal
        if len(mentions) >= limit:
            break
    return mentions


def analyze_research_gaps(papers: List[dict]) -> dict:
    """
    Main entry point. Returns a structured dict (never a single unstructured
    string) with clearly separated observed evidence vs. hypotheses.
    """
    if not papers:
        return {
            "num_papers_analyzed": 0,
            "note": "No papers were supplied for gap analysis. Run a search first.",
            "frequent_topics": [],
            "underrepresented_topics": [],
            "missing_combinations": [],
            "methodological_limitations": [],
            "contradictory_findings": [],
            "future_work_suggestions": [],
        }

    frequent = _top_topics(papers)
    underrepresented = _underrepresented_topics(papers)
    missing_combos = _missing_combinations(frequent, papers)
    limitations = _find_pattern_mentions(papers, _LIMITATION_RE, field="abstract")
    contradictions = _find_pattern_mentions(papers, _CONTRADICTION_RE, field="abstract")

    future_work = [
        {
            "paper_id": p.get("paper_id", ""),
            "title": p.get("title", ""),
            "suggestion": p.get("tldr") or "See abstract for future-work discussion.",
            "evidence_type": "observed",
        }
        for p in papers if re.search(r"future work|future research", p.get("abstract", ""), re.IGNORECASE)
    ][:10]

    return {
        "num_papers_analyzed": len(papers),
        "frequent_topics": frequent,
        "underrepresented_topics": underrepresented,
        "missing_combinations": missing_combos,
        "methodological_limitations": limitations,
        "contradictory_findings": contradictions,
        "future_work_suggestions": future_work,
        "disclaimer": (
            "Items marked evidence_type='observed' are derived directly from "
            "the supplied papers. Items marked 'hypothesis' (e.g. missing "
            "topic combinations) are potential, unverified research "
            "directions -- not established facts."
        ),
    }
