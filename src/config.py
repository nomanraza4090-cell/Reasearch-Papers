"""
config.py
=========
Central configuration for the ResearchMind project.

Every path, model name, hyperparameter, and API setting used anywhere in the
project should be imported from this module rather than hard-coded elsewhere.
Secrets (API keys) are loaded from environment variables via python-dotenv
and are never hard-coded.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv is optional at import time (e.g. inside minimal test
    # environments). Environment variables set another way still work.
    pass


# ---------------------------------------------------------------------------
# Base directories
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

MODELS_DIR = BASE_DIR / "models"
VECTOR_DB_DIR = BASE_DIR / "vector_db" / "faiss_index"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
EVALUATIONS_DIR = OUTPUTS_DIR / "evaluations"

# ---------------------------------------------------------------------------
# Dataset file paths
# ---------------------------------------------------------------------------
RAW_CSV_PATH = RAW_DATA_DIR / "research_papers_raw.csv"
CLEAN_CSV_PATH = PROCESSED_DATA_DIR / "research_papers_clean.csv"
FINAL_CSV_PATH = PROCESSED_DATA_DIR / "research_papers_final.csv"
CHUNKS_JSON_PATH = PROCESSED_DATA_DIR / "paper_chunks.json"

EMBEDDINGS_PATH = EMBEDDINGS_DIR / "paper_embeddings.npy"
EMBEDDINGS_METADATA_PATH = EMBEDDINGS_DIR / "metadata.json"

FAISS_INDEX_PATH = VECTOR_DB_DIR / "index.faiss"
FAISS_METADATA_PATH = VECTOR_DB_DIR / "metadata.json"

RANKING_MODEL_PATH = MODELS_DIR / "ranking" / "paper_ranker.pkl"

# ---------------------------------------------------------------------------
# Columns the pipeline understands. None of these are guaranteed to be
# present in every dataset -- all downstream code must check for presence
# before use (see src/data_loader.py: get_available_columns).
# ---------------------------------------------------------------------------
KNOWN_COLUMNS = [
    "paperId", "paper_id",
    "title",
    "abstract",
    "authors",
    "year",
    "venue",
    "url",
    "citationCount", "citation_count",
    "referenceCount", "reference_count",
    "publicationDate", "publication_date",
    "fieldsOfStudy", "fields_of_study",
    "influential_citation_count",
    "tldr",
    "research_topic",
]

# Canonical column names used internally, mapped from the many possible
# raw-dataset spellings (camelCase from the Semantic Scholar API, or
# snake_case from a scraped/exported CSV).
COLUMN_ALIASES = {
    "paper_id": ["paperId", "paper_id", "id"],
    "title": ["title"],
    "abstract": ["abstract"],
    "authors": ["authors"],
    "year": ["year"],
    "venue": ["venue"],
    "url": ["url"],
    "citation_count": ["citationCount", "citation_count"],
    "reference_count": ["referenceCount", "reference_count"],
    "publication_date": ["publicationDate", "publication_date"],
    "fields_of_study": ["fieldsOfStudy", "fields_of_study"],
    "influential_citation_count": ["influential_citation_count"],
    "tldr": ["tldr"],
    "research_topic": ["research_topic"],
}

REQUIRED_MINIMUM_COLUMNS = ["title", "abstract"]

# ---------------------------------------------------------------------------
# Embedding configuration
# ---------------------------------------------------------------------------
# Primary, required-by-spec embedding model.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-mpnet-base-v2")
EMBEDDING_DIM_MPNET = 768

# ResearchMind must never crash or fabricate results if heavy ML
# dependencies (sentence-transformers / torch) are not installed in the
# current environment. When they are unavailable, the system automatically
# falls back to a transparent TF-IDF + SVD embedding backend, and this is
# always logged clearly (never silently). The backend actually used is
# recorded in embeddings metadata so results are never presented as if they
# came from a model that didn't actually run.
EMBEDDING_BACKEND_OVERRIDE: Optional[str] = os.getenv("EMBEDDING_BACKEND")  # "sentence-transformers" | "tfidf" | None (auto)
TFIDF_FALLBACK_DIM = 384

EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
NORMALIZE_EMBEDDINGS = True

# ---------------------------------------------------------------------------
# Chunking configuration (used for long-form text such as abstracts/TLDRs
# when constructing RAG context windows)
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))       # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # characters

# ---------------------------------------------------------------------------
# Vector store configuration
# ---------------------------------------------------------------------------
# "cosine" is implemented as inner-product search over L2-normalized
# vectors, which is mathematically equivalent and is FAISS's recommended
# approach for cosine similarity search.
FAISS_METRIC = "cosine"
VECTOR_BACKEND_OVERRIDE: Optional[str] = os.getenv("VECTOR_BACKEND")  # "faiss" | "numpy" | None (auto)

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))

# ---------------------------------------------------------------------------
# Ranking configuration
# ---------------------------------------------------------------------------
RANKING_WEIGHTS = {
    "semantic": float(os.getenv("RANK_W_SEMANTIC", "0.65")),
    "citation": float(os.getenv("RANK_W_CITATION", "0.15")),
    "recency": float(os.getenv("RANK_W_RECENCY", "0.10")),
    "quality": float(os.getenv("RANK_W_QUALITY", "0.10")),
}
assert abs(sum(RANKING_WEIGHTS.values()) - 1.0) < 1e-6, "RANKING_WEIGHTS must sum to 1.0"

RECENCY_HALF_LIFE_YEARS = 8  # years after which recency score decays to ~50%

# ---------------------------------------------------------------------------
# LLM / RAG configuration
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # "anthropic" | "openai" | "none"
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
LLM_API_KEY = os.getenv("LLM_API_KEY")  # never printed, never hard-coded
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "6000"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def ensure_directories() -> None:
    """Create every directory this project relies on, if missing."""
    for d in [
        RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR,
        MODELS_DIR / "embedding_model", MODELS_DIR / "classifier", MODELS_DIR / "ranking",
        VECTOR_DB_DIR, FIGURES_DIR, REPORTS_DIR, EVALUATIONS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def get_logger(name: str):
    """Return a module-level logger configured with LOG_FORMAT/LOG_LEVEL."""
    import logging

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
        logger.propagate = False
    return logger
