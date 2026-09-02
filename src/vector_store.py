"""
vector_store.py
================
FAISS-backed vector database for similarity search, with a numpy
exact-search fallback when the `faiss` package is not installed in the
current environment.

The numpy fallback is exact cosine-similarity search (a real, correct
algorithm -- not a stub) and is perfectly adequate at the "1,000 to
100,000 papers" scale this project targets; FAISS is preferred for larger
scale and lower latency and is the default whenever it is available.

The active backend is always recorded in the saved index metadata so
downstream code and users can see exactly what ran.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src import config

logger = config.get_logger(__name__)


class VectorStoreError(Exception):
    pass


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
        return True
    except ImportError:
        return False


def resolve_backend() -> str:
    override = config.VECTOR_BACKEND_OVERRIDE
    if override:
        if override == "faiss" and not _faiss_available():
            raise VectorStoreError(
                "VECTOR_BACKEND=faiss was requested but the `faiss` package "
                "is not installed. Fix: pip install faiss-cpu."
            )
        return override
    if _faiss_available():
        return "faiss"
    logger.warning(
        "faiss is not installed in this environment. Falling back to an "
        "exact numpy cosine-similarity search. Install faiss-cpu for "
        "better performance at scale. See requirements.txt / README.md."
    )
    return "numpy"


class VectorStore:
    """
    Thin wrapper unifying a FAISS index and a numpy fallback behind one
    interface: build / add / search / save / load / validate.
    """

    def __init__(self, dim: int, backend: Optional[str] = None):
        self.dim = dim
        self.backend = backend or resolve_backend()
        self._faiss_index = None
        self._numpy_matrix: Optional[np.ndarray] = None
        self.metadata: List[dict] = []

    # -- construction ------------------------------------------------------

    def build(self, embeddings: np.ndarray, metadata: List[dict]) -> None:
        if embeddings.shape[0] != len(metadata):
            raise VectorStoreError(
                f"Cannot build index: {embeddings.shape[0]} embeddings vs "
                f"{len(metadata)} metadata records. These must match exactly."
            )
        embeddings = embeddings.astype(np.float32)

        if self.backend == "faiss":
            import faiss
            index = faiss.IndexFlatIP(self.dim)  # inner product == cosine on normalized vectors
            index.add(embeddings)
            self._faiss_index = index
        else:
            self._numpy_matrix = embeddings.copy()

        self.metadata = list(metadata)
        logger.info("Built %s-backed vector store with %s vectors (dim=%s)", self.backend, len(metadata), self.dim)

    # -- search --------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return a list of (index, similarity_score) pairs, best first."""
        if self.backend == "faiss" and self._faiss_index is None:
            raise VectorStoreError("Vector store has not been built or loaded yet.")
        if self.backend == "numpy" and self._numpy_matrix is None:
            raise VectorStoreError("Vector store has not been built or loaded yet.")

        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        top_k = min(top_k, len(self.metadata))
        if top_k <= 0:
            return []

        if self.backend == "faiss":
            scores, indices = self._faiss_index.search(query_vector, top_k)
            return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i != -1]
        else:
            sims = (self._numpy_matrix @ query_vector.T).flatten()
            top_idx = np.argpartition(-sims, min(top_k, len(sims) - 1))[:top_k]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
            return [(int(i), float(sims[i])) for i in top_idx]

    # -- persistence ---------------------------------------------------

    def save(self, index_path: Path = None, metadata_path: Path = None) -> None:
        index_path = index_path or config.FAISS_INDEX_PATH
        metadata_path = metadata_path or config.FAISS_METADATA_PATH
        index_path.parent.mkdir(parents=True, exist_ok=True)

        if self.backend == "faiss":
            import faiss
            faiss.write_index(self._faiss_index, str(index_path))
        else:
            # Store the raw matrix alongside using a sibling .npy file so a
            # numpy-backed "index" can also be reloaded exactly.
            np.save(str(index_path) + ".numpy.npy", self._numpy_matrix)
            index_path.write_text("NUMPY_FALLBACK_INDEX")  # placeholder marker file

        payload = {
            "backend": self.backend,
            "dim": self.dim,
            "num_vectors": len(self.metadata),
            "metric": config.FAISS_METRIC,
            "records": self.metadata,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info("Saved %s-backed index (%s vectors) to %s", self.backend, len(self.metadata), index_path)

    @classmethod
    def load(cls, index_path: Path = None, metadata_path: Path = None) -> "VectorStore":
        index_path = index_path or config.FAISS_INDEX_PATH
        metadata_path = metadata_path or config.FAISS_METADATA_PATH

        if not index_path.exists() or not metadata_path.exists():
            raise VectorStoreError(
                "FAISS index not found.\n\n"
                f"Expected:\n  {index_path}\n  {metadata_path}\n\n"
                "Run notebooks/05_Vector_Database.ipynb (or "
                "`python -m src.vector_store`) before starting semantic search."
            )

        with open(metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        backend = payload["backend"]
        store = cls(dim=payload["dim"], backend=backend)
        store.metadata = payload["records"]

        if backend == "faiss":
            if not _faiss_available():
                raise VectorStoreError(
                    "This index was built with the faiss backend, but faiss is "
                    "not installed in the current environment. "
                    "Fix: pip install faiss-cpu, or rebuild the index with "
                    "VECTOR_BACKEND=numpy."
                )
            import faiss
            store._faiss_index = faiss.read_index(str(index_path))
            num_vectors = store._faiss_index.ntotal
        else:
            store._numpy_matrix = np.load(str(index_path) + ".numpy.npy")
            num_vectors = store._numpy_matrix.shape[0]

        if num_vectors != len(store.metadata):
            raise VectorStoreError(
                f"Data integrity error: index has {num_vectors} vectors but "
                f"metadata has {len(store.metadata)} records. Rebuild the index."
            )

        logger.info("Loaded %s-backed index with %s vectors from %s", backend, num_vectors, index_path)
        return store

    def validate(self) -> None:
        n_vectors = self._faiss_index.ntotal if self.backend == "faiss" else self._numpy_matrix.shape[0]
        if n_vectors != len(self.metadata):
            raise VectorStoreError(
                f"FAISS/vector count ({n_vectors}) does not match metadata "
                f"count ({len(self.metadata)})."
            )


def build_and_save_index(embeddings: np.ndarray, metadata: List[dict]) -> VectorStore:
    """Convenience helper used by notebook 05 and by run.py bootstrapping."""
    store = VectorStore(dim=embeddings.shape[1])
    store.build(embeddings, metadata)
    store.validate()
    store.save()
    return store
