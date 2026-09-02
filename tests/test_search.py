import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import embeddings as emb_mod
from src import vector_store as vs_mod
from src import semantic_search


@pytest.fixture()
def small_index(tmp_path, monkeypatch):
    """Build a tiny, isolated vector index so tests don't depend on the
    real project data/embeddings being present."""
    emb_mod._model_cache.clear()

    texts = [
        "Transformer models for medical image segmentation.",
        "A survey of reinforcement learning algorithms.",
        "Graph neural networks for molecule property prediction.",
        "Large language models for clinical text summarization.",
    ]
    metadata = [
        {"paper_id": str(i), "title": t, "authors": "A. Author", "year": 2020 + i,
         "venue": "Test Venue", "url": "https://example.com", "citation_count": i * 10,
         "abstract": t, "search_text": t}
        for i, t in enumerate(texts)
    ]
    vectors = emb_mod.embed_documents(texts)

    index_path = tmp_path / "index.faiss"
    metadata_path = tmp_path / "metadata.json"
    store = vs_mod.VectorStore(dim=vectors.shape[1], backend=vs_mod.resolve_backend())
    store.build(vectors, metadata)
    store.save(index_path, metadata_path)

    monkeypatch.setattr(semantic_search, "_get_store", lambda: store)
    yield store


def test_search_papers_returns_results(small_index):
    results = semantic_search.search_papers("medical imaging with transformers", top_k=2)
    assert len(results) == 2


def test_search_results_contain_required_fields(small_index):
    results = semantic_search.search_papers("reinforcement learning", top_k=1)
    required = {"rank", "paper_id", "title", "authors", "year", "venue", "abstract", "url", "similarity_score"}
    assert required.issubset(results[0].keys())


def test_search_results_have_similarity_scores(small_index):
    results = semantic_search.search_papers("graph neural networks", top_k=3)
    for r in results:
        assert isinstance(r["similarity_score"], float)


def test_top_k_parameter_respected(small_index):
    results = semantic_search.search_papers("clinical text", top_k=1)
    assert len(results) == 1
    results = semantic_search.search_papers("clinical text", top_k=4)
    assert len(results) == 4


def test_search_raises_on_empty_query(small_index):
    with pytest.raises(ValueError):
        semantic_search.search_papers("")
