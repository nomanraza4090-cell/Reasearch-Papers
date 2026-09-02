import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ranking


def make_paper(**kwargs):
    base = {
        "title": "Sample Paper", "similarity_score": 0.5, "citation_count": 10,
        "year": 2022, "venue": "Test Venue", "authors": "A. Author",
        "abstract": " ".join(["word"] * 60), "url": "https://example.com",
    }
    base.update(kwargs)
    return base


def test_compute_scores_returns_sorted_results():
    papers = [make_paper(similarity_score=0.2), make_paper(similarity_score=0.9)]
    scored = ranking.compute_scores(papers)
    assert scored[0]["final_score"] >= scored[1]["final_score"]


def test_compute_scores_values_are_valid_range():
    papers = [make_paper()]
    scored = ranking.compute_scores(papers)
    for key in ("semantic_score", "citation_score", "recency_score", "quality_score"):
        assert 0.0 <= scored[0][key] <= 1.0


def test_compute_scores_handles_missing_metadata_without_crashing():
    papers = [{"title": "Minimal Paper", "similarity_score": 0.4}]  # no citation_count/year/etc.
    scored = ranking.compute_scores(papers)
    assert scored[0]["final_score"] >= 0.0


def test_compute_scores_rejects_bad_weights():
    with pytest.raises(ValueError):
        ranking.compute_scores([make_paper()], weights={"semantic": 0.5, "citation": 0.5, "recency": 0.5, "quality": 0.5})


def test_recency_score_decays_with_age():
    recent = ranking.recency_score(ranking.CURRENT_YEAR)
    old = ranking.recency_score(ranking.CURRENT_YEAR - 40)
    assert recent > old


def test_train_learned_ranker_rejects_empty_data():
    with pytest.raises(ValueError):
        ranking.train_learned_ranker([], [])


def test_load_learned_ranker_raises_clear_error_when_missing(tmp_path):
    missing_path = tmp_path / "no_such_model.pkl"
    with pytest.raises(ranking.LearnedRankerUnavailable):
        ranking.load_learned_ranker(missing_path)
