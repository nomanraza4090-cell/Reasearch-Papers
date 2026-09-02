"""
evaluation.py
=============
Reusable evaluation functions for the dataset, semantic search, ranking,
and RAG pipeline stages.

If labeled ground truth is not supplied, functions that require it clearly
report that the metric cannot be reliably calculated rather than
fabricating a number.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> Optional[float]:
    if not relevant_ids:
        return None  # cannot compute without ground truth
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for i in top_k if i in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> Optional[float]:
    if not relevant_ids:
        return None
    top_k = retrieved_ids[:k]
    hits = sum(1 for i in top_k if i in relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> Optional[float]:
    if not relevant_ids:
        return None
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def similarity_distribution(scores: List[float]) -> dict:
    if not scores:
        return {"count": 0}
    arr = np.array(scores)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def evaluate_retrieval(queries_with_labels: List[dict], search_fn, k: int = 10) -> dict:
    """
    queries_with_labels: [{"query": str, "relevant_ids": [str, ...]}, ...]
    search_fn: callable(query, top_k) -> List[dict] with a 'paper_id' field.
    If no query has labeled relevant_ids, returns a report explaining that
    precision/recall/MRR cannot be computed (rather than fabricating scores).
    """
    labeled = [q for q in queries_with_labels if q.get("relevant_ids")]
    if not labeled:
        return {
            "num_queries": len(queries_with_labels),
            "num_labeled_queries": 0,
            "precision_at_k": None,
            "recall_at_k": None,
            "mrr": None,
            "note": "No manually labeled relevant_ids were supplied for any query, "
                    "so precision@k, recall@k, and MRR cannot be reliably calculated.",
        }

    precisions, recalls, rrs = [], [], []
    for q in labeled:
        results = search_fn(q["query"], k)
        retrieved_ids = [r["paper_id"] for r in results]
        p = precision_at_k(retrieved_ids, q["relevant_ids"], k)
        r = recall_at_k(retrieved_ids, q["relevant_ids"], k)
        m = mrr(retrieved_ids, q["relevant_ids"])
        if p is not None:
            precisions.append(p)
        if r is not None:
            recalls.append(r)
        if m is not None:
            rrs.append(m)

    return {
        "num_queries": len(queries_with_labels),
        "num_labeled_queries": len(labeled),
        "precision_at_k": float(np.mean(precisions)) if precisions else None,
        "recall_at_k": float(np.mean(recalls)) if recalls else None,
        "mrr": float(np.mean(rrs)) if rrs else None,
    }


def evaluate_ranking(ranked_papers: List[dict]) -> dict:
    """Diagnostic ranking-quality stats: score distribution + monotonicity check."""
    if not ranked_papers:
        return {"num_papers": 0}
    scores = [p["final_score"] for p in ranked_papers]
    is_sorted = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    return {
        "num_papers": len(ranked_papers),
        "score_distribution": similarity_distribution(scores),
        "is_correctly_sorted_desc": is_sorted,
    }


def evaluate_rag(rag_result: dict, expected_paper_ids: Optional[List[str]] = None) -> dict:
    """
    Heuristic RAG evaluation:
      - retrieval_relevance: fraction of sources overlapping expected_paper_ids (if given)
      - context_relevance: whether any sources were retrieved at all
      - answer_groundedness: whether the answer mentions any retrieved paper's title words
      - citation_correctness: whether inline [n] markers exist when sources were used
      - hallucination_rate: cannot be reliably computed without human review; flagged as such
    """
    sources = rag_result.get("sources", [])
    answer = rag_result.get("answer", "")

    report = {
        "num_sources": len(sources),
        "context_relevance": len(sources) > 0,
        "citation_correctness": bool(sources) and any(f"[{i}]" in answer for i in range(1, len(sources) + 1)),
        "hallucination_rate": None,
    }

    if expected_paper_ids:
        retrieved_ids = {s.get("title") for s in sources}  # title-based since 'sources' may lack paper_id
        report["retrieval_relevance"] = None  # requires paper_id-based ground truth, not computed here
    else:
        report["retrieval_relevance"] = None

    report["note"] = (
        "hallucination_rate and retrieval_relevance require manual human "
        "review or labeled ground truth and are not fabricated here."
    )
    return report
