# ResearchMind

**ResearchMind** is an AI-powered research paper discovery and analysis platform. It ingests a corpus of research papers, cleans and analyzes them, generates semantic embeddings, stores them in a vector database, and exposes semantic search, paper ranking, paper comparison, research-gap analysis, literature-review generation, and a RAG-powered conversational chatbot — all through a Streamlit web interface.

---

## 1. Project Overview

ResearchMind covers the full pipeline from raw scraped/exported paper metadata to an interactive research assistant:

```
Raw Dataset → Cleaning → Deduplication → Text Analysis → Embeddings →
FAISS Vector DB → Semantic Search → Ranking → Gap Analysis / Comparison →
Literature Review / RAG → Chatbot → Streamlit UI
```

Every stage is implemented as a reusable, independently testable module in `src/`, demonstrated in a numbered notebook in `notebooks/`, and surfaced in the Streamlit app in `app/`.

## 2. Architecture

```
                    ResearchMind
                         │
                         ▼
               ┌──────────────────┐
               │ Research Dataset │
               └────────┬─────────┘
                        ▼
               ┌──────────────────┐
               │ Preprocessing    │
               └────────┬─────────┘
                        ▼
               ┌──────────────────┐
               │ Text Analysis    │
               └────────┬─────────┘
                        ▼
               ┌──────────────────┐
               │ MPNet Embeddings │
               └────────┬─────────┘
                        ▼
               ┌──────────────────┐
               │ FAISS Vector DB  │
               └────────┬─────────┘
                        ▼
               ┌──────────────────┐
               │ Semantic Search  │
               └────────┬─────────┘
             ┌──────────┼──────────┐
             ▼          ▼          ▼
          Ranking   Comparison   Gap Analysis
             │          │          │
             └──────────┼──────────┘
                        ▼
               ┌──────────────────┐
               │ Literature       │
               │ Review / RAG     │
               └────────┬─────────┘
                        ▼
               ┌──────────────────┐
               │ Research Chatbot │
               └────────┬─────────┘
                        ▼
               ┌──────────────────┐
               │    Streamlit     │
               │   Web Interface  │
               └──────────────────┘
```

### A note on optional heavy dependencies (read this first)

ResearchMind's primary, spec-required stack is **`sentence-transformers` (`all-mpnet-base-v2`)** for embeddings and **FAISS** for vector search. Both are optional, heavy dependencies (the former needs `torch` and downloads ~420MB of model weights on first use; the latter needs a compiled native extension).

If either package is not installed in the current environment, ResearchMind **does not crash and does not fabricate results**. Instead it:

1. Logs a clear, visible warning explaining exactly what's missing and how to fix it.
2. Falls back to a real, correct, transparent alternative:
   - **Embeddings:** TF-IDF (scikit-learn) + Truncated SVD dimensionality reduction, L2-normalized.
   - **Vector search:** exact numpy cosine-similarity search (correct at this project's target scale of 1K–100K papers).
3. Records which backend actually ran in `data/embeddings/metadata.json` (`embedding_backend`) and `vector_db/faiss_index/metadata.json` (`backend`), so results are never silently presented as if the primary model had run.

**To use the real MPNet + FAISS stack:** `pip install -r requirements.txt` in an environment with internet access, then re-run notebooks 04 and 05. No code changes are required — the backend is auto-detected.

## 3. Features

- **Data exploration & cleaning** — dynamic column detection, null/duplicate handling, quality filtering
- **Text analysis** — word frequency, n-grams, TF-IDF keywords, publication/citation trends
- **Semantic embeddings** — `all-mpnet-base-v2` (with documented fallback)
- **FAISS vector database** — cosine similarity search (with documented fallback)
- **Semantic search** — meaning-based, not keyword matching; configurable top-K
- **Paper ranking** — transparent, configurable weighted scoring (semantic + citations + recency + quality); optional learned ranker
- **Paper comparison** — structured, extractive comparison across objective/methodology/dataset/results/strengths/limitations/future work
- **Research gap analysis** — frequent vs. underrepresented topics, missing concept combinations, limitations, contradictions; observed evidence vs. hypotheses always labeled separately
- **Literature review generation** — themed, referenced, extractive-by-default
- **RAG pipeline** — configurable LLM provider, grounded + cited answers, graceful extractive fallback with no LLM key
- **Research chatbot** — conversational, context-aware, routes to search/ranking/comparison/gap-analysis/lit-review/RAG
- **Evaluation suite** — dataset, retrieval (precision@k/recall@k/MRR), ranking, and RAG diagnostics; never fabricates metrics it can't compute
- **Streamlit UI** — dashboard, search, ranking, comparison, gap analysis, literature review, chatbot

## 4. Dataset Structure

Primary dataset: `data/raw/research_papers_raw.csv`

ResearchMind dynamically detects available columns via `src/data_loader.py::get_available_columns` and **never crashes** because an optional column is missing. Recognized fields (any common alias, camelCase or snake_case):

| Canonical field | Aliases |
|---|---|
| `paper_id` | `paperId`, `paper_id`, `id` |
| `title` | `title` |
| `abstract` | `abstract` |
| `authors` | `authors` |
| `year` | `year` |
| `venue` | `venue` |
| `url` | `url` |
| `citation_count` | `citationCount`, `citation_count` |
| `reference_count` | `referenceCount`, `reference_count` |
| `publication_date` | `publicationDate`, `publication_date` |
| `fields_of_study` | `fieldsOfStudy`, `fields_of_study` |

Only `title` and `abstract` (under any alias) are strictly required; every other field is optional.

## 5. Installation

```bash
git clone <your-repo-url>
cd ResearchMind
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 6. Virtual Environment Setup

Using `venv` (shown above) is recommended. `conda` works equally well:

```bash
conda create -n researchmind python=3.11
conda activate researchmind
pip install -r requirements.txt
```

## 7. Environment Variables

```bash
cp .env.example .env
```

Then edit `.env` and fill in real values as needed:

```text
SEMANTIC_SCHOLAR_API_KEY=your_key_here
LLM_API_KEY=your_key_here
LLM_PROVIDER=anthropic          # "anthropic" | "openai" | "none"
```

`.env` is gitignored and must never be committed. No API key is ever hard-coded or printed anywhere in the codebase.

## 8. Notebook Execution Order

Run these in order — each depends on the previous stage's output:

```
01_Data_Exploration        (reads raw CSV, no modification)
02_Data_Preprocessing      (produces research_papers_clean.csv, research_papers_final.csv)
03_Text_Analysis           (produces figures in outputs/figures/)
04_Embedding_Generation    (produces paper_embeddings.npy, metadata.json)
05_Vector_Database         (produces vector_db/faiss_index/{index.faiss,metadata.json})
06_Semantic_Search         (demonstrates src/semantic_search.py)
07_Paper_Ranking           (demonstrates src/ranking.py)
08_Research_Gap_Analysis   (produces outputs/reports/research_gap_report.json)
09_RAG_Pipeline            (demonstrates src/rag.py)
10_Evaluation              (produces outputs/evaluations/evaluation_report.json)
```

Each notebook documents its own `INPUT` / `PROCESSING` / `OUTPUT` in its first markdown cell.

## 9. Building Embeddings

```bash
jupyter nbconvert --to notebook --execute notebooks/04_Embedding_Generation.ipynb
```

or run the equivalent from Python:

```python
from src import data_loader, embeddings as emb_mod
df = data_loader.load_final_dataset()
vectors = emb_mod.embed_documents(df["search_text"].tolist())
```

## 10. Building FAISS

```bash
jupyter nbconvert --to notebook --execute notebooks/05_Vector_Database.ipynb
```

This validates `number of vectors == number of metadata records` and raises a clear error if they ever desynchronize.

## 11. Running Semantic Search

```python
from src.semantic_search import search_papers
results = search_papers("transformer models for medical imaging", top_k=10)
```

## 12. Running Streamlit

```bash
python run.py
# or directly:
streamlit run app/streamlit_app.py
```

`run.py` validates the full pipeline (directories, files, embeddings, FAISS index) before launching and refuses to launch with a clear explanation if something is missing.

## 13. Running Tests

```bash
pytest tests/ -v
```

Tests do not require external APIs or internet access; the embedding/vector-store fallbacks (see section 2) make the full suite runnable offline.

## 14. Project Structure

```text
ResearchMind/
├── data/{raw,processed,embeddings}/
├── notebooks/01..10_*.ipynb
├── models/{embedding_model,classifier,ranking}/
├── vector_db/faiss_index/
├── src/                  # all reusable business logic
├── app/                  # Streamlit UI (streamlit_app.py + components/ + assets/)
├── tests/
├── outputs/{figures,reports,evaluations}/
├── .env / .env.example / .gitignore
├── requirements.txt
├── run.py
└── README.md
```

## 15. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FAISS index not found` | Notebook 05 hasn't run | Run `05_Vector_Database.ipynb` |
| `Embedding count does not match metadata count` | Embeddings/metadata desynced | Regenerate both together via notebook 04 |
| `sentence-transformers is not installed... falling back to TF-IDF` | Optional heavy dependency missing | `pip install sentence-transformers torch` |
| `faiss is not installed... falling back to numpy` | Optional native dependency missing | `pip install faiss-cpu` |
| RAG answers are extractive, not generated | No `LLM_API_KEY` configured | Set `LLM_API_KEY` and `LLM_PROVIDER` in `.env` |
| `precision_at_k` / `recall_at_k` / `mrr` return `None` | No labeled relevance judgments supplied | Populate `labeled_queries` in notebook 10 with real ground truth |

## 16. Deployment Instructions

1. Provision a host with Python 3.10+ (Streamlit Community Cloud, a VM, or a container).
2. Install dependencies: `pip install -r requirements.txt`.
3. Either commit the generated `data/processed/`, `data/embeddings/`, and `vector_db/` artifacts (see `.gitignore` for how to opt into this), or run notebooks 02/04/05 as a build step.
4. Set environment variables (`LLM_API_KEY`, etc.) via your host's secrets manager — never commit `.env`.
5. Start with `streamlit run app/streamlit_app.py` (or `python run.py`).

## 17. Limitations

- The transparent TF-IDF/numpy fallbacks are weaker than MPNet/FAISS at capturing deep semantic relationships; install the full stack for production-quality search.
- Research-gap and paper-comparison analysis are extractive/heuristic (keyword and pattern-based), not a substitute for expert literature review.
- Retrieval/ranking evaluation metrics (precision@k, recall@k, MRR) require manually labeled relevance judgments that are not included by default.
- The optional learned ranking model is never trained on fabricated labels — it activates only once real labeled data is supplied.

## 18. Future Improvements

- Add a supervised SciBERT classifier once a labeled classification task/dataset is defined.
- Add cross-encoder re-ranking as a second-stage refinement over FAISS candidates.
- Add streaming token-by-token responses in the chatbot UI when using a streaming-capable LLM provider.
- Add citation-graph-based analysis (co-citation, bibliographic coupling) for deeper research-gap detection.
- Add multi-lingual embedding support for non-English corpora.
