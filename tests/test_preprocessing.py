import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import preprocessing


def test_clean_text_removes_control_chars_and_whitespace():
    dirty = "Hello \x00\x1f  World  \n\t "
    assert preprocessing.clean_text(dirty) == "Hello World"


def test_clean_text_handles_none_and_nan():
    assert preprocessing.clean_text(None) == ""
    assert preprocessing.clean_text(float("nan")) == ""


def test_normalize_title_strips_punctuation_and_lowercases():
    assert preprocessing.normalize_title("A Study: Of AI!!") == "a study of ai"


def test_normalize_year_valid_and_invalid():
    assert preprocessing.normalize_year(2023) == 2023
    assert preprocessing.normalize_year("2023") == 2023
    assert preprocessing.normalize_year("not a year") is None
    assert preprocessing.normalize_year(1500) is None


def test_has_useful_text():
    assert preprocessing.has_useful_text("A real title", "A" * 40) is True
    assert preprocessing.has_useful_text("", "A" * 40) is False
    assert preprocessing.has_useful_text("Title", "short") is False


def test_remove_null_rows_drops_fully_empty():
    df = pd.DataFrame({
        "title": ["Real Title", "", None],
        "abstract": ["", "Real abstract text here.", None],
    })
    col_map = {"title": "title", "abstract": "abstract"}
    out = preprocessing.remove_null_rows(df, col_map)
    assert len(out) == 2  # only the fully-null row is dropped


def test_deduplicate_papers_by_normalized_title():
    df = pd.DataFrame({
        "paper_id": ["1", "2"],
        "title": ["A Study of AI", "a study of ai!!"],
        "abstract": ["abstract one", "abstract two"],
    })
    col_map = {"paper_id": "paper_id", "title": "title"}
    out = preprocessing.deduplicate_papers(df, col_map)
    assert len(out) == 1


def test_build_search_text_combines_fields():
    df = pd.DataFrame({
        "title": ["My Title"],
        "abstract": ["My abstract."],
        "venue": ["My Venue"],
        "fields_of_study": ["Computer Science"],
    })
    out = preprocessing.build_search_text(df)
    text = out.loc[0, "search_text"]
    assert "My Title" in text and "My abstract." in text and "My Venue" in text


def test_quality_filter_removes_short_records():
    df = pd.DataFrame({
        "title": ["Long Enough Title", "Hi"],
        "abstract": [" ".join(["word"] * 30), "too short"],
    })
    out = preprocessing.quality_filter(df, min_abstract_words=20, min_title_words=2)
    assert len(out) == 1


def test_clean_dataset_never_crashes_on_missing_optional_columns():
    df = pd.DataFrame({
        "title": ["A Valid Title"],
        "abstract": [" ".join(["word"] * 25)],
    })
    out = preprocessing.clean_dataset(df)
    assert "search_text" in out.columns
    assert len(out) == 1
