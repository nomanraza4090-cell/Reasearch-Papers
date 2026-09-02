#!/usr/bin/env python3
"""
run.py
======
ResearchMind entry point.

Validates the project's directory structure and required pipeline
artifacts, prints a clear, actionable status report, and (if everything is
ready) launches the Streamlit application.

Usage:
    python run.py            # validate, then launch Streamlit if ready
    python run.py --check    # validate only, do not launch
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config


def _status_line(ok: bool, label: str) -> str:
    icon = "✅" if ok else "❌"
    return f"  {icon} {label}"


def validate_directories() -> list[str]:
    problems = []
    required_dirs = [
        config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR, config.EMBEDDINGS_DIR,
        config.VECTOR_DB_DIR, config.FIGURES_DIR, config.REPORTS_DIR, config.EVALUATIONS_DIR,
    ]
    for d in required_dirs:
        if not d.exists():
            problems.append(f"Missing directory: {d}")
    return problems


def validate_files() -> dict:
    checks = {
        "Raw dataset": config.RAW_CSV_PATH,
        "Clean dataset": config.CLEAN_CSV_PATH,
        "Final dataset": config.FINAL_CSV_PATH,
        "Embeddings (.npy)": config.EMBEDDINGS_PATH,
        "Embeddings metadata": config.EMBEDDINGS_METADATA_PATH,
        "FAISS/vector index": config.FAISS_INDEX_PATH,
        "Vector index metadata": config.FAISS_METADATA_PATH,
    }
    return {label: path.exists() for label, path in checks.items()}


def validate_faiss_index() -> tuple[bool, str]:
    try:
        from src import vector_store as vs_mod
        store = vs_mod.VectorStore.load()
        store.validate()
        return True, f"Vector index OK ({len(store.metadata)} vectors, backend={store.backend})"
    except Exception as e:
        return False, str(e)


def validate_embeddings() -> tuple[bool, str]:
    try:
        from src import embeddings as emb_mod
        vectors, records, payload = emb_mod.load_embeddings()
        return True, f"Embeddings OK ({vectors.shape[0]} vectors, dim={vectors.shape[1]}, backend={payload.get('embedding_backend')})"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="ResearchMind entry point")
    parser.add_argument("--check", action="store_true", help="Validate only, do not launch Streamlit")
    args = parser.parse_args()

    print("=" * 60)
    print("ResearchMind — Startup Validation")
    print("=" * 60)

    config.ensure_directories()

    dir_problems = validate_directories()
    if dir_problems:
        print("\nDirectory issues (auto-created where possible):")
        for p in dir_problems:
            print(f"  - {p}")

    file_checks = validate_files()
    print("\nRequired files:")
    for label, ok in file_checks.items():
        print(_status_line(ok, label))

    all_files_ok = all(file_checks.values())

    embeddings_ok, embeddings_msg = (False, "Skipped (embeddings.npy missing)")
    index_ok, index_msg = (False, "Skipped (index missing)")
    if file_checks["Embeddings (.npy)"] and file_checks["Embeddings metadata"]:
        embeddings_ok, embeddings_msg = validate_embeddings()
    if file_checks["FAISS/vector index"] and file_checks["Vector index metadata"]:
        index_ok, index_msg = validate_faiss_index()

    print("\nPipeline integrity:")
    print(_status_line(embeddings_ok, embeddings_msg))
    print(_status_line(index_ok, index_msg))

    ready = all_files_ok and embeddings_ok and index_ok

    print("\n" + "=" * 60)
    if ready:
        print("✅ ResearchMind is ready.")
    else:
        print("❌ ResearchMind is NOT fully set up yet.")
        print(
            "\nRun the notebooks in order to build missing artifacts:\n"
            "  01_Data_Exploration -> 02_Data_Preprocessing -> 03_Text_Analysis ->\n"
            "  04_Embedding_Generation -> 05_Vector_Database -> ... -> 10_Evaluation\n"
            "\nSee README.md, section 'Notebook Execution Order', for details."
        )
    print("=" * 60)

    if args.check:
        sys.exit(0 if ready else 1)

    if not ready:
        print("\nRefusing to launch the Streamlit app until the pipeline is ready.")
        sys.exit(1)

    print("\nLaunching Streamlit app...\n")
    app_path = Path(__file__).resolve().parent / "app" / "streamlit_app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


if __name__ == "__main__":
    main()
