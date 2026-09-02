import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import rag


SAMPLE_PAPERS = [
    {"paper_id": "1", "title": "Transformers for Medical Imaging", "authors": "A. Author",
     "year": 2022, "venue": "Test Venue", "url": "https://example.com/1",
     "abstract": "This paper studies transformers applied to medical imaging tasks.",
     "similarity_score": 0.8, "citation_count": 20},
    {"paper_id": "2", "title": "Survey of Deep Learning in Healthcare", "authors": "B. Author",
     "year": 2021, "venue": "Test Venue", "url": "https://example.com/2",
     "abstract": "A broad survey of deep learning techniques used across healthcare applications.",
     "similarity_score": 0.6, "citation_count": 50},
]


def test_construct_context_includes_citation_markers():
    context = rag.construct_context(SAMPLE_PAPERS)
    assert "[1]" in context and "[2]" in context
    assert "Transformers for Medical Imaging" in context


def test_construct_context_respects_max_chars():
    context = rag.construct_context(SAMPLE_PAPERS, max_chars=50)
    assert len(context) <= 60  # allow small overhead for separators


def test_extractive_fallback_answer_used_without_llm_key(monkeypatch):
    monkeypatch.setattr(rag.config, "LLM_API_KEY", None)
    monkeypatch.setattr(rag, "retrieve", lambda q, top_k=None: SAMPLE_PAPERS)
    result = rag.generate_answer("What methods are used for medical imaging?")
    assert result["used_llm"] is False
    assert "extractive" in result["answer"].lower() or "retrieved" in result["answer"].lower()
    assert len(result["sources"]) == 2


def test_generate_answer_handles_empty_retrieval(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda q, top_k=None: [])
    result = rag.generate_answer("A question with no matching papers")
    assert result["sources"] == []
    assert "no relevant papers" in result["answer"].lower()


def test_generate_answer_preserves_citations_in_sources(monkeypatch):
    monkeypatch.setattr(rag.config, "LLM_API_KEY", None)
    monkeypatch.setattr(rag, "retrieve", lambda q, top_k=None: SAMPLE_PAPERS)
    result = rag.generate_answer("What datasets were used?")
    titles = [s["title"] for s in result["sources"]]
    assert "Transformers for Medical Imaging" in titles
    assert "Survey of Deep Learning in Healthcare" in titles
