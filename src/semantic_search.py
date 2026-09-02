"""
semantic_search.py
===================
User Query -> Query Embedding -> FAISS/numpy Search -> Top-K Papers ->
Formatted Results.
"""

from __future__ import annotations

from typing import List, Optional

from src import config, embeddings as emb_mod, vector_store as vs_mod

logger = config.get_logger(__name__)

_store_cache: Optional[vs_mod.VectorStore] = None


def _get_store() -> vs_mod.VectorStore:
    global _store_cache
    if _store_cache is None:
        _store_cache = vs_mod.VectorStore.load()
        # If the corpus was embedded with the TF-IDF fallback backend, the
        # fitted vectorizer/SVD lives only in-process and must be re-fit
        # against the persisted corpus text after a fresh process start so
        # that embed_query() produces vectors in the same space as the
        # stored document vectors. load_embeddings() handles this refit.
        try:
            emb_mod.load_embeddings()
        except FileNotFoundError:
            pass  # embeddings.npy may not be needed if store was built in-memory only
    return _store_cache


def reset_cache() -> None:
    """Force the next search to reload the index from disk (e.g. after a rebuild)."""
    global _store_cache
    _store_cache = None


def search_papers(query: str, top_k: int = None) -> List[dict]:
    """
    Perform semantic search over the paper corpus.

    Returns a list of dicts, each containing:
        rank, paper_id, title, authors, year, venue, abstract, url,
        similarity_score
    ordered by descending similarity.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    top_k = top_k or config.DEFAULT_TOP_K
    store = _get_store()

    query_vector = emb_mod.embed_query(query)
    hits = store.search(query_vector, top_k=top_k)

    results = []
    for rank, (idx, score) in enumerate(hits, start=1):
        record = store.metadata[idx]
        results.append({
            "rank": rank,
            "index": idx,
            "paper_id": record.get("paper_id", ""),
            "title": record.get("title", ""),
            "authors": record.get("authors", ""),
            "year": record.get("year"),
            "venue": record.get("venue", ""),
            "abstract": record.get("abstract", ""),
            "url": record.get("url", ""),
            "citation_count": record.get("citation_count", 0),
            "similarity_score": round(float(score), 4),
        })
    return results
