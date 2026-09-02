"""
embeddings.py
=============
Embedding-model loading, query/document embedding, normalization, and
persistence.

Backend selection
------------------
The project's primary, spec-required backend is `sentence-transformers`
with the `all-mpnet-base-v2` model. That library (and its `torch`
dependency) is a heavy, optional install. ResearchMind must never crash or
fabricate fake embeddings if it is missing -- instead it clearly logs a
fallback to a transparent TF-IDF + Truncated-SVD embedding backend built on
scikit-learn (already a required dependency).

The backend actually used for a given run is always recorded in
`data/embeddings/metadata.json` (`embedding_backend` field) so results are
never presented as if they came from a model that did not actually run.
Switch on purpose with the `EMBEDDING_BACKEND` environment variable
("sentence-transformers" or "tfidf").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from src import config

logger = config.get_logger(__name__)

_model_cache = {}


class EmbeddingBackendError(Exception):
    pass


def _sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def resolve_backend() -> str:
    override = config.EMBEDDING_BACKEND_OVERRIDE
    if override:
        if override == "sentence-transformers" and not _sentence_transformers_available():
            raise EmbeddingBackendError(
                "EMBEDDING_BACKEND=sentence-transformers was requested but the "
                "`sentence-transformers` package (and its `torch` dependency) is "
                "not installed.\n"
                "Fix: pip install -r requirements.txt (in an environment with "
                "internet access to download packages and the model weights)."
            )
        return override
    if _sentence_transformers_available():
        return "sentence-transformers"
    logger.warning(
        "sentence-transformers is not installed in this environment. "
        "Falling back to a TF-IDF + SVD embedding backend so the rest of "
        "the pipeline remains runnable. For production-quality semantic "
        "embeddings (the project's primary design target), install "
        "sentence-transformers + torch and re-run embedding generation. "
        "See requirements.txt / README.md."
    )
    return "tfidf"


# ---------------------------------------------------------------------------
# sentence-transformers backend
# ---------------------------------------------------------------------------

def _load_sentence_transformer_model(model_name: str = None):
    model_name = model_name or config.EMBEDDING_MODEL_NAME
    cache_key = f"st::{model_name}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise EmbeddingBackendError(
            "sentence-transformers is not installed. Run: pip install sentence-transformers torch"
        ) from e
    logger.info("Loading sentence-transformers model '%s' (this may download weights on first use)...", model_name)
    model = SentenceTransformer(model_name)
    _model_cache[cache_key] = model
    return model


def _embed_sentence_transformers(texts: Sequence[str], batch_size: int, model_name: str = None) -> np.ndarray:
    model = _load_sentence_transformer_model(model_name)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
    )
    return np.asarray(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# TF-IDF fallback backend (deterministic, no downloads, no GPU/torch needed)
# ---------------------------------------------------------------------------

def _get_or_fit_tfidf_pipeline(corpus: Sequence[str]):
    """
    Fit (or retrieve a cached) TF-IDF vectorizer + SVD reducer on `corpus`.
    Cached in-process so repeated calls (e.g. document embedding, then
    query embedding) reuse the same fitted vocabulary/projection.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    cache_key = "tfidf_pipeline"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    if not corpus:
        raise EmbeddingBackendError("Cannot fit the TF-IDF fallback embedder on an empty corpus.")

    vectorizer = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)

    n_components = min(config.TFIDF_FALLBACK_DIM, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
    n_components = max(n_components, 2)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(tfidf_matrix)

    pipeline = {"vectorizer": vectorizer, "svd": svd}
    _model_cache[cache_key] = pipeline
    logger.info(
        "Fitted TF-IDF fallback embedder: vocab=%s, reduced_dim=%s",
        len(vectorizer.vocabulary_), n_components,
    )
    return pipeline


def _embed_tfidf(texts: Sequence[str], fit_corpus: Optional[Sequence[str]] = None) -> np.ndarray:
    pipeline = _get_or_fit_tfidf_pipeline(fit_corpus if fit_corpus is not None else texts)
    tfidf_matrix = pipeline["vectorizer"].transform(texts)
    reduced = pipeline["svd"].transform(tfidf_matrix).astype(np.float32)
    if config.NORMALIZE_EMBEDDINGS:
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        reduced = reduced / norms
    return reduced


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_documents(
    texts: Sequence[str],
    batch_size: int = None,
    backend: Optional[str] = None,
) -> np.ndarray:
    """
    Embed a batch of documents. Returns an (N, D) float32 numpy array.
    Handles empty/failed inputs safely rather than crashing the whole batch.
    """
    batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
    backend = backend or resolve_backend()

    texts = [t if isinstance(t, str) and t.strip() else " " for t in texts]  # avoid empty-string failures

    if not texts:
        dim = config.EMBEDDING_DIM_MPNET if backend == "sentence-transformers" else config.TFIDF_FALLBACK_DIM
        return np.zeros((0, dim), dtype=np.float32)

    try:
        if backend == "sentence-transformers":
            return _embed_sentence_transformers(texts, batch_size)
        elif backend == "tfidf":
            return _embed_tfidf(texts)
        else:
            raise EmbeddingBackendError(f"Unknown embedding backend: {backend}")
    except EmbeddingBackendError:
        raise
    except Exception as e:
        raise EmbeddingBackendError(f"Embedding generation failed using backend '{backend}': {e}") from e


def embed_query(query: str, backend: Optional[str] = None) -> np.ndarray:
    """Embed a single query string. Returns a 1-D float32 array."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    backend = backend or resolve_backend()

    if backend == "tfidf" and "tfidf_pipeline" not in _model_cache:
        raise EmbeddingBackendError(
            "The TF-IDF fallback embedder has not been fit yet. "
            "Run document embedding (embed_documents over the corpus) before "
            "embedding queries, or load a persisted embedding session."
        )

    vec = embed_documents([query], batch_size=1, backend=backend)
    return vec[0]


def save_embeddings(
    embeddings: np.ndarray,
    metadata: List[dict],
    backend: str,
    embeddings_path: Path = None,
    metadata_path: Path = None,
) -> None:
    embeddings_path = embeddings_path or config.EMBEDDINGS_PATH
    metadata_path = metadata_path or config.EMBEDDINGS_METADATA_PATH

    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"Embedding count ({embeddings.shape[0]}) does not match metadata "
            f"count ({len(metadata)}). Refusing to save a desynchronized pair."
        )

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)

    payload = {
        "embedding_backend": backend,
        "embedding_model": config.EMBEDDING_MODEL_NAME if backend == "sentence-transformers" else "tfidf-svd-fallback",
        "embedding_dim": int(embeddings.shape[1]),
        "num_records": len(metadata),
        "records": metadata,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    logger.info(
        "Saved %s embeddings (dim=%s, backend=%s) to %s and metadata to %s",
        embeddings.shape[0], embeddings.shape[1], backend, embeddings_path, metadata_path,
    )


def load_embeddings(embeddings_path: Path = None, metadata_path: Path = None):
    embeddings_path = embeddings_path or config.EMBEDDINGS_PATH
    metadata_path = metadata_path or config.EMBEDDINGS_METADATA_PATH

    if not embeddings_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Embeddings not found.\n"
            f"Expected:\n  {embeddings_path}\n  {metadata_path}\n\n"
            "Run notebooks/04_Embedding_Generation.ipynb before continuing."
        )

    embeddings = np.load(embeddings_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    records = payload.get("records", payload if isinstance(payload, list) else [])
    if embeddings.shape[0] != len(records):
        raise ValueError(
            f"Data integrity error: {embeddings.shape[0]} embeddings but "
            f"{len(records)} metadata records. The embeddings and metadata "
            "files are desynchronized -- regenerate both together."
        )

    # If this was a TF-IDF-backend embedding set, the fitted vectorizer/SVD
    # is NOT persisted (by design -- it is a lightweight fallback, not a
    # deployed artifact). Re-fit it against the loaded corpus so embed_query
    # works consistently after a fresh process start.
    backend = payload.get("embedding_backend", "sentence-transformers")
    if backend == "tfidf" and "tfidf_pipeline" not in _model_cache:
        corpus = [r.get("search_text") or r.get("title", "") for r in records]
        _get_or_fit_tfidf_pipeline(corpus)

    return embeddings, records, payload
