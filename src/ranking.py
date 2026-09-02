"""
ranking.py
==========
Weighted, configurable paper-ranking system combining semantic relevance
with citation count, publication recency, and metadata quality.

final_score = semantic * w_semantic + citation * w_citation
              + recency * w_recency + quality * w_quality

Weights come from config.RANKING_WEIGHTS and always sum to 1.0.

A learned ranking model is supported (see train_learned_ranker /
load_learned_ranker) but is only used when explicitly requested and when
labeled training data is actually supplied -- ResearchMind does not
fabricate a fake ML model just to produce the paper_ranker.pkl filename.
"""

from __future__ import annotations

import math
import pickle
from datetime import date
from pathlib import Path
from typing import List, Optional

import numpy as np

from src import config

logger = config.get_logger(__name__)

CURRENT_YEAR = date.today().year


def normalize_semantic_score(similarity: float) -> float:
    """Similarity scores from cosine search are already in [-1, 1]; clip to [0, 1]."""
    return float(max(0.0, min(1.0, (similarity + 1.0) / 2.0 if similarity < 0 else similarity)))


def normalize_citation_score(citation_count: Optional[int], corpus_max: int) -> float:
    """Log-scaled citation score in [0, 1], robust to a few extremely highly-cited outliers."""
    if not citation_count or citation_count <= 0:
        return 0.0
    if corpus_max <= 0:
        return 0.0
    return float(math.log1p(citation_count) / math.log1p(max(corpus_max, 1)))


def recency_score(year: Optional[int], half_life_years: int = config.RECENCY_HALF_LIFE_YEARS) -> float:
    """Exponential decay: a paper published `half_life_years` ago scores ~0.5."""
    if not year:
        return 0.0
    try:
        age = max(0, CURRENT_YEAR - int(year))
    except (TypeError, ValueError):
        return 0.0
    return float(0.5 ** (age / half_life_years))


def quality_score(paper: dict) -> float:
    """
    Metadata-completeness heuristic in [0, 1]: rewards papers with a venue,
    authors, a reasonably substantial abstract, and a valid URL. This is a
    transparent proxy, not a claim about scientific quality.
    """
    score = 0.0
    weight_each = 1.0 / 4
    if paper.get("venue"):
        score += weight_each
    if paper.get("authors"):
        score += weight_each
    abstract = paper.get("abstract") or ""
    if len(str(abstract).split()) >= 50:
        score += weight_each
    if paper.get("url", "").startswith("http"):
        score += weight_each
    return float(score)


def compute_scores(
    papers: List[dict],
    weights: Optional[dict] = None,
) -> List[dict]:
    """
    Given search results (each must include 'similarity_score', optionally
    'citation_count' and 'year'), compute per-paper component scores and a
    combined final_score. Returns a new list, sorted by final_score desc.
    """
    weights = weights or config.RANKING_WEIGHTS
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError(f"Ranking weights must sum to 1.0, got {sum(weights.values())}: {weights}")

    corpus_max_citations = max((p.get("citation_count") or 0 for p in papers), default=0)

    scored = []
    for paper in papers:
        sem = normalize_semantic_score(paper.get("similarity_score", 0.0))
        cit = normalize_citation_score(paper.get("citation_count"), corpus_max_citations)
        rec = recency_score(paper.get("year"))
        qual = quality_score(paper)

        final = (
            sem * weights["semantic"]
            + cit * weights["citation"]
            + rec * weights["recency"]
            + qual * weights["quality"]
        )

        enriched = dict(paper)
        enriched.update({
            "semantic_score": round(sem, 4),
            "citation_score": round(cit, 4),
            "recency_score": round(rec, 4),
            "quality_score": round(qual, 4),
            "final_score": round(float(final), 4),
        })
        scored.append(enriched)

    scored.sort(key=lambda p: p["final_score"], reverse=True)
    for i, p in enumerate(scored, start=1):
        p["rank"] = i
    return scored


# ---------------------------------------------------------------------------
# Optional learned ranker
# ---------------------------------------------------------------------------

class LearnedRankerUnavailable(Exception):
    pass


def train_learned_ranker(training_rows: List[dict], labels: List[float], path: Path = None):
    """
    Train a simple learned ranker (gradient-boosted regression over the
    same feature set as the weighted scorer) IF labeled relevance data is
    supplied. Refuses to run with empty/insufficient data rather than
    fabricating a model.
    """
    path = path or config.RANKING_MODEL_PATH
    if not training_rows or not labels or len(training_rows) != len(labels):
        raise ValueError(
            "train_learned_ranker requires non-empty, equal-length "
            "`training_rows` and `labels` (e.g. human relevance judgments). "
            "No labeled data was supplied, so no learned ranker will be "
            "trained or saved. The transparent weighted-scoring approach in "
            "compute_scores() remains the default ranking strategy."
        )
    from sklearn.ensemble import GradientBoostingRegressor

    X = np.array([
        [
            normalize_semantic_score(r.get("similarity_score", 0.0)),
            normalize_citation_score(r.get("citation_count"), max((rr.get("citation_count") or 0) for rr in training_rows) or 1),
            recency_score(r.get("year")),
            quality_score(r),
        ]
        for r in training_rows
    ])
    y = np.array(labels)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "feature_order": ["semantic", "citation", "recency", "quality"]}, f)
    logger.info("Trained and saved learned ranker to %s", path)
    return model


def load_learned_ranker(path: Path = None):
    path = path or config.RANKING_MODEL_PATH
    if not path.exists():
        raise LearnedRankerUnavailable(
            f"No learned ranker found at {path}. Falling back to the "
            "transparent weighted-scoring approach (compute_scores())."
        )
    with open(path, "rb") as f:
        return pickle.load(f)
