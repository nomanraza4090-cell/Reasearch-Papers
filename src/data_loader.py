"""
data_loader.py
===============
Reusable, dependency-light functions for loading and validating the
ResearchMind paper dataset at any stage of the pipeline (raw, clean, final).

Design principle: never assume a column exists. Always check first, and
fail with a clear, actionable error message rather than a bare traceback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from src import config

logger = config.get_logger(__name__)


class DataLoadError(Exception):
    """Raised when a dataset cannot be loaded or fails validation."""


def _missing_file_message(path: Path, hint: str) -> str:
    return (
        f"Required file not found:\n"
        f"  {path}\n\n"
        f"{hint}"
    )


def resolve_column(df: pd.DataFrame, canonical_name: str) -> Optional[str]:
    """
    Return the actual column name present in `df` that corresponds to a
    canonical field (e.g. 'citation_count'), or None if no matching column
    exists under any known alias.
    """
    aliases = config.COLUMN_ALIASES.get(canonical_name, [canonical_name])
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


def get_available_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Map every canonical column ResearchMind knows about to the actual
    column name present in `df` (or None if absent). Downstream code should
    use this mapping instead of hard-coding column names.
    """
    return {canonical: resolve_column(df, canonical) for canonical in config.COLUMN_ALIASES}


def validate_schema(df: pd.DataFrame, required: Optional[List[str]] = None) -> Dict[str, Optional[str]]:
    """
    Validate that a dataframe has at least the minimum columns ResearchMind
    needs to function (by default: title + abstract, under any known alias).

    Returns the resolved column mapping on success.
    Raises DataLoadError with a clear explanation on failure.
    """
    required = required or config.REQUIRED_MINIMUM_COLUMNS
    mapping = get_available_columns(df)

    missing = [c for c in required if mapping.get(c) is None]
    if missing:
        raise DataLoadError(
            "Dataset is missing required column(s): "
            f"{missing}.\n"
            f"Columns present in dataset: {list(df.columns)}\n"
            f"ResearchMind recognizes these aliases per field: "
            f"{ {c: config.COLUMN_ALIASES[c] for c in missing} }"
        )

    optional_missing = [c for c, v in mapping.items() if v is None and c not in required]
    if optional_missing:
        logger.info(
            "Optional columns not present in this dataset (pipeline will "
            "handle this gracefully): %s", optional_missing
        )

    return mapping


def load_raw_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw, untouched dataset. Never mutate or overwrite this file."""
    path = path or config.RAW_CSV_PATH
    if not path.exists():
        raise DataLoadError(_missing_file_message(
            path,
            "Place the scraped/exported research-paper CSV at this location "
            "before running the pipeline (see README.md, section 'Dataset')."
        ))
    df = pd.read_csv(path)
    logger.info("Loaded raw dataset: %s rows, %s columns from %s", len(df), len(df.columns), path)
    validate_schema(df)
    return df


def load_clean_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or config.CLEAN_CSV_PATH
    if not path.exists():
        raise DataLoadError(_missing_file_message(
            path,
            "The cleaned dataset has not been generated yet.\n"
            "Run notebooks/02_Data_Preprocessing.ipynb (or "
            "`python -m src.preprocessing`) before continuing."
        ))
    df = pd.read_csv(path)
    logger.info("Loaded clean dataset: %s rows from %s", len(df), path)
    return df


def load_final_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or config.FINAL_CSV_PATH
    if not path.exists():
        raise DataLoadError(_missing_file_message(
            path,
            "The final, quality-filtered dataset has not been generated yet.\n"
            "Run notebooks/02_Data_Preprocessing.ipynb through to completion."
        ))
    df = pd.read_csv(path)
    logger.info("Loaded final dataset: %s rows from %s", len(df), path)
    return df


def load_metadata(path: Optional[Path] = None) -> List[dict]:
    """Load embedding-index -> paper-metadata mapping (a list, index-aligned)."""
    path = path or config.EMBEDDINGS_METADATA_PATH
    if not path.exists():
        raise DataLoadError(_missing_file_message(
            path,
            "Embedding metadata has not been generated yet.\n"
            "Run notebooks/04_Embedding_Generation.ipynb first."
        ))
    with open(path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    logger.info("Loaded %s metadata records from %s", len(metadata), path)
    return metadata


def load_paper_records(df: Optional[pd.DataFrame] = None) -> List[dict]:
    """Return the dataset as a list of plain dicts (paper records)."""
    if df is None:
        df = load_final_dataset()
    return df.to_dict(orient="records")


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %s rows to %s", len(df), path)
