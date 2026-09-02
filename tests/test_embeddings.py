import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import embeddings as emb_mod


@pytest.fixture(autouse=True)
def clear_model_cache():
    emb_mod._model_cache.clear()
    yield
    emb_mod._model_cache.clear()


def test_resolve_backend_returns_known_value():
    backend = emb_mod.resolve_backend()
    assert backend in ("sentence-transformers", "tfidf")


def test_embed_documents_correct_dimensions():
    texts = [
        "Deep learning for medical imaging.",
        "Transformer models in natural language processing.",
        "A survey of reinforcement learning methods.",
    ]
    vectors = emb_mod.embed_documents(texts)
    assert vectors.shape[0] == len(texts)
    assert vectors.shape[1] > 0
    assert vectors.dtype == np.float32


def test_embed_documents_non_empty_vectors():
    texts = ["Some research text about AI."]
    vectors = emb_mod.embed_documents(texts)
    assert not np.allclose(vectors, 0)


def test_embed_documents_handles_empty_list():
    vectors = emb_mod.embed_documents([])
    assert vectors.shape[0] == 0


def test_embed_query_consistent_dimension_with_documents():
    texts = ["Paper about graph neural networks.", "Paper about computer vision."]
    doc_vectors = emb_mod.embed_documents(texts)
    query_vector = emb_mod.embed_query("graph neural networks")
    assert query_vector.shape[0] == doc_vectors.shape[1]


def test_save_and_load_embeddings_round_trip(tmp_path):
    texts = ["Paper A about NLP.", "Paper B about vision."]
    vectors = emb_mod.embed_documents(texts)
    metadata = [{"paper_id": "1", "title": "Paper A", "search_text": texts[0]},
                {"paper_id": "2", "title": "Paper B", "search_text": texts[1]}]

    emb_path = tmp_path / "embeddings.npy"
    meta_path = tmp_path / "metadata.json"
    emb_mod.save_embeddings(vectors, metadata, backend=emb_mod.resolve_backend(),
                             embeddings_path=emb_path, metadata_path=meta_path)

    loaded_vectors, loaded_records, payload = emb_mod.load_embeddings(emb_path, meta_path)
    assert loaded_vectors.shape == vectors.shape
    assert len(loaded_records) == 2
    assert payload["num_records"] == 2


def test_save_embeddings_raises_on_count_mismatch(tmp_path):
    vectors = np.zeros((3, 8), dtype=np.float32)
    metadata = [{"paper_id": "1"}]  # mismatched count
    with pytest.raises(ValueError):
        emb_mod.save_embeddings(vectors, metadata, backend="tfidf",
                                 embeddings_path=tmp_path / "e.npy",
                                 metadata_path=tmp_path / "m.json")
