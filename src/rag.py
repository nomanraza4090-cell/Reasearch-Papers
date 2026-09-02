"""
rag.py
======
User Question -> Query Embedding -> FAISS Retrieval -> Context Construction
-> LLM -> Grounded Answer.

Independent of Streamlit. LLM provider is configurable via
config.LLM_PROVIDER / config.LLM_MODEL, and the API key is read only from
the environment (.env), never hard-coded or printed.

If no LLM API key is configured, `generate_answer` still returns a
grounded, purely extractive answer built directly from retrieved papers
(clearly labeled as such) rather than crashing or fabricating a response.
"""

from __future__ import annotations

import os
from typing import List, Optional

from src import config, semantic_search

logger = config.get_logger(__name__)


class RAGError(Exception):
    pass


def retrieve(query: str, top_k: int = None) -> List[dict]:
    top_k = top_k or config.RAG_TOP_K
    try:
        return semantic_search.search_papers(query, top_k=top_k)
    except FileNotFoundError as e:
        raise RAGError(str(e)) from e


def construct_context(papers: List[dict], max_chars: int = None) -> str:
    """
    Build an LLM-ready context block from retrieved papers, each tagged
    with a citation marker [n] so the LLM (or the extractive fallback) can
    reference sources unambiguously and avoid hallucinating paper details.
    """
    max_chars = max_chars or config.RAG_MAX_CONTEXT_CHARS
    blocks = []
    used_chars = 0
    for i, p in enumerate(papers, start=1):
        block = (
            f"[{i}] Title: {p.get('title','Untitled')}\n"
            f"Authors: {p.get('authors','Unknown')} | Year: {p.get('year','n.d.')} | Venue: {p.get('venue','')}\n"
            f"Abstract: {(p.get('abstract') or '')[:800]}\n"
        )
        if used_chars + len(block) > max_chars:
            break
        blocks.append(block)
        used_chars += len(block)
    return "\n---\n".join(blocks)


def _build_prompt(question: str, context: str) -> str:
    return (
        "You are a research assistant. Answer the question using ONLY the "
        "numbered sources below. Cite sources inline using [n]. If the "
        "sources do not contain the answer, say so explicitly instead of "
        "guessing. Do not invent paper titles, authors, or findings that "
        "are not present in the sources.\n\n"
        f"Sources:\n{context}\n\n"
        f"Question: {question}\n\nGrounded answer:"
    )


def _call_llm(prompt: str) -> str:
    """
    Provider-agnostic LLM call. Returns None if no provider/key is
    configured, so callers can fall back to an extractive answer.
    """
    if config.LLM_PROVIDER == "none" or not config.LLM_API_KEY:
        logger.info("No LLM API key configured (LLM_API_KEY unset) -- using extractive fallback answer.")
        return None

    if config.LLM_PROVIDER == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise RAGError(
                "LLM_PROVIDER=anthropic but the `anthropic` package is not "
                "installed. Fix: pip install anthropic."
            )
        client = anthropic.Anthropic(api_key=config.LLM_API_KEY)
        response = client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")

    elif config.LLM_PROVIDER == "openai":
        try:
            import openai
        except ImportError:
            raise RAGError(
                "LLM_PROVIDER=openai but the `openai` package is not "
                "installed. Fix: pip install openai."
            )
        client = openai.OpenAI(api_key=config.LLM_API_KEY)
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    raise RAGError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")


def _extractive_fallback_answer(question: str, papers: List[dict]) -> str:
    """A transparent, no-LLM-required grounded answer built from abstracts/tldrs."""
    if not papers:
        return "No relevant papers were retrieved for this question."
    lines = [f"Based on {len(papers)} retrieved paper(s):"]
    for i, p in enumerate(papers, start=1):
        summary = p.get("tldr") or (p.get("abstract", "")[:220] + "...")
        lines.append(f"[{i}] {p.get('title','Untitled')} ({p.get('year','n.d.')}): {summary}")
    lines.append(
        "\n(No LLM provider is configured -- this is an extractive summary "
        "of retrieved sources, not a generated narrative. Set LLM_API_KEY "
        "in .env to enable generated, grounded answers.)"
    )
    return "\n".join(lines)


def generate_answer(question: str, top_k: int = None) -> dict:
    """
    Full RAG call. Returns:
        {
          "question": ...,
          "answer": ...,
          "sources": [ {index, title, authors, year, url}, ... ],
          "used_llm": bool,
        }
    """
    papers = retrieve(question, top_k=top_k)

    if not papers:
        return {
            "question": question,
            "answer": "No relevant papers were found in the index for this question. Try rephrasing or broadening your query.",
            "sources": [],
            "used_llm": False,
        }

    context = construct_context(papers)
    prompt = _build_prompt(question, context)

    llm_answer = None
    try:
        llm_answer = _call_llm(prompt)
    except RAGError as e:
        logger.warning("LLM call failed, using extractive fallback: %s", e)

    used_llm = llm_answer is not None
    answer = llm_answer if used_llm else _extractive_fallback_answer(question, papers)

    sources = [
        {"index": i, "title": p["title"], "authors": p["authors"], "year": p["year"], "url": p["url"]}
        for i, p in enumerate(papers, start=1)
    ]

    return {"question": question, "answer": answer, "sources": sources, "used_llm": used_llm}
