"""
preprocessing.py
=================
All text-cleaning / dataset-cleaning logic used by
notebooks/02_Data_Preprocessing.ipynb. Every function here is independently
testable (see tests/test_preprocessing.py) and has no Streamlit or notebook
dependency.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

import pandas as pd

from src import config
from src.data_loader import get_available_columns, resolve_column

logger = config.get_logger(__name__)

_URL_RE = re.compile(r"^(https?://)[^\s]+$", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ---------------------------------------------------------------------------
# Low-level text cleaning primitives
# ---------------------------------------------------------------------------

def remove_control_characters(text: str) -> str:
    if not isinstance(text, str):
        return text
    return _CONTROL_CHARS_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    if not isinstance(text, str):
        return text
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_text(text: Optional[str]) -> str:
    """General-purpose cleaner used for both titles and abstracts."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = remove_control_characters(text)
    text = normalize_whitespace(text)
    return text


def normalize_title(title: Optional[str]) -> str:
    """Lowercased, punctuation-normalized title used for deduplication."""
    cleaned = clean_text(title).lower()
    cleaned = re.sub(r"[^a-z0-9\s]", "", cleaned)
    cleaned = normalize_whitespace(cleaned)
    return cleaned


def clean_abstract(abstract: Optional[str]) -> str:
    return clean_text(abstract)


def normalize_authors(authors: Optional[str]) -> str:
    """
    Normalize an author string. Accepts either a comma-separated string or
    something already clean, and returns a clean, comma-and-space joined
    string with duplicate whitespace removed.
    """
    if authors is None or (isinstance(authors, float) and pd.isna(authors)):
        return ""
    authors = clean_text(str(authors))
    parts = [normalize_whitespace(p) for p in authors.split(",")]
    parts = [p for p in parts if p]
    return ", ".join(parts)


def normalize_year(year) -> Optional[int]:
    """Coerce a year value to a plausible 4-digit int, or None if invalid."""
    try:
        y = int(float(year))
    except (TypeError, ValueError):
        return None
    if 1900 <= y <= 2100:
        return y
    return None


def is_valid_url(url: Optional[str]) -> bool:
    if not isinstance(url, str) or not url.strip():
        return False
    return bool(_URL_RE.match(url.strip()))


# ---------------------------------------------------------------------------
# Row-level validity checks
# ---------------------------------------------------------------------------

def has_useful_text(title: str, abstract: str, min_abstract_len: int = 30) -> bool:
    """A paper without a usable title+abstract is not useful for semantic search."""
    title_ok = isinstance(title, str) and len(title.strip()) >= 3
    abstract_ok = isinstance(abstract, str) and len(abstract.strip()) >= min_abstract_len
    return title_ok and abstract_ok


def is_malformed_record(row: pd.Series, col_map: dict) -> bool:
    """Detect obviously broken rows (e.g. title == abstract, both empty, etc.)."""
    title = clean_text(row.get(col_map.get("title"), ""))
    abstract = clean_text(row.get(col_map.get("abstract"), ""))
    if not title and not abstract:
        return True
    if title and abstract and title.strip().lower() == abstract.strip().lower():
        return True
    return False


# ---------------------------------------------------------------------------
# Dataset-level pipeline steps
# ---------------------------------------------------------------------------

def remove_null_rows(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Drop rows that are null/empty on both title and abstract."""
    title_col = col_map.get("title")
    abstract_col = col_map.get("abstract")
    before = len(df)

    def _keep(row):
        title = row.get(title_col, "") if title_col else ""
        abstract = row.get(abstract_col, "") if abstract_col else ""
        return not (pd.isna(title) or str(title).strip() == "") or not (pd.isna(abstract) or str(abstract).strip() == "")

    df = df[df.apply(_keep, axis=1)].copy()
    logger.info("remove_null_rows: %s -> %s rows", before, len(df))
    return df


def deduplicate_papers(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """
    Deduplicate using, in order of preference: paper_id, then normalized
    title. Keeps the first occurrence (assumes the dataset is not
    pre-sorted by quality; ranking/quality filtering happens later).
    """
    before = len(df)
    id_col = col_map.get("paper_id")
    title_col = col_map.get("title")

    if id_col and id_col in df.columns:
        df = df.drop_duplicates(subset=[id_col], keep="first")

    if title_col and title_col in df.columns:
        df["_normalized_title_dedup"] = df[title_col].apply(normalize_title)
        df = df.drop_duplicates(subset=["_normalized_title_dedup"], keep="first")
        df = df.drop(columns=["_normalized_title_dedup"])

    logger.info("deduplicate_papers: %s -> %s rows", before, len(df))
    return df.reset_index(drop=True)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pass corresponding to notebooks/02_Data_Preprocessing.ipynb,
    stage 1 ("clean" dataset, before final quality filtering).
    """
    col_map = get_available_columns(df)
    df = df.copy()

    df = remove_null_rows(df, col_map)

    if col_map.get("title"):
        df[col_map["title"]] = df[col_map["title"]].apply(clean_text)
    if col_map.get("abstract"):
        df[col_map["abstract"]] = df[col_map["abstract"]].apply(clean_abstract)
    if col_map.get("authors"):
        df[col_map["authors"]] = df[col_map["authors"]].apply(normalize_authors)
    if col_map.get("year"):
        df[col_map["year"]] = df[col_map["year"]].apply(normalize_year)
    if col_map.get("citation_count"):
        df[col_map["citation_count"]] = pd.to_numeric(df[col_map["citation_count"]], errors="coerce").fillna(0).astype(int)
    if col_map.get("reference_count"):
        df[col_map["reference_count"]] = pd.to_numeric(df[col_map["reference_count"]], errors="coerce").fillna(0).astype(int)
    if col_map.get("url"):
        df["_url_valid"] = df[col_map["url"]].apply(is_valid_url)

    before = len(df)
    df = df[~df.apply(lambda r: is_malformed_record(r, col_map), axis=1)].copy()
    logger.info("remove_malformed_records: %s -> %s rows", before, len(df))

    df = deduplicate_papers(df, col_map)

    before = len(df)
    title_col, abstract_col = col_map.get("title"), col_map.get("abstract")
    df = df[df.apply(
        lambda r: has_useful_text(r.get(title_col, ""), r.get(abstract_col, "")), axis=1
    )].copy()
    logger.info("remove_without_useful_text: %s -> %s rows", before, len(df))

    df = build_search_text(df)
    return df.reset_index(drop=True)


def build_search_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a single combined `search_text` field:
        title + abstract + fields_of_study + venue
    used for embeddings / keyword search / TF-IDF.
    """
    col_map = get_available_columns(df)
    df = df.copy()

    def _row_text(row) -> str:
        parts = []
        for canonical in ("title", "abstract", "fields_of_study", "venue"):
            col = col_map.get(canonical)
            if col and col in row and pd.notna(row[col]):
                parts.append(str(row[col]))
        return normalize_whitespace(" ".join(parts))

    df["search_text"] = df.apply(_row_text, axis=1)
    return df


def quality_filter(
    df: pd.DataFrame,
    min_abstract_words: int = 20,
    min_title_words: int = 2,
) -> pd.DataFrame:
    """
    Final quality-filtering stage that produces research_papers_final.csv.
    Removes remaining low-quality records (too-short abstracts/titles).
    """
    col_map = get_available_columns(df)
    title_col, abstract_col = col_map.get("title"), col_map.get("abstract")
    before = len(df)

    def _ok(row) -> bool:
        title_words = len(str(row.get(title_col, "")).split())
        abstract_words = len(str(row.get(abstract_col, "")).split())
        return title_words >= min_title_words and abstract_words >= min_abstract_words

    df = df[df.apply(_ok, axis=1)].copy()
    logger.info("quality_filter: %s -> %s rows", before, len(df))
    return df.reset_index(drop=True)


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Lightweight dataset-integrity report. Used by tests and by the
    evaluation notebook.
    """
    col_map = get_available_columns(df)
    report = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns_present": {k: v for k, v in col_map.items() if v is not None},
        "columns_missing": [k for k, v in col_map.items() if v is None],
        "has_search_text": "search_text" in df.columns,
    }
    if col_map.get("title"):
        report["duplicate_titles"] = int(
            df[col_map["title"]].apply(normalize_title).duplicated().sum()
        )
    return report
