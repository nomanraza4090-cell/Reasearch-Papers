"""
chatbot.py
==========
Reusable, UI-independent research chatbot with conversational context.
Streamlit (or any other UI) drives this class; it never imports Streamlit.

Supported intents (regex/keyword-routed, and always falls through to RAG
for anything not explicitly matched):
    - "find/search papers about X"          -> semantic_search
    - "which paper is most relevant"        -> ranking over last results
    - "compare the top two papers"          -> paper_comparison
    - "what datasets were commonly used"    -> extractive scan of last results
    - "what research gaps exist"            -> research_gap
    - "future research directions"          -> research_gap (future_work)
    - "literature review on X"              -> literature_review
    - anything else                          -> rag.generate_answer
"""

from __future__ import annotations

import re
from typing import List, Optional

from src import config, semantic_search, ranking, paper_comparison, research_gap, literature_review, rag

logger = config.get_logger(__name__)


class ResearchChatbot:
    def __init__(self, top_k: int = None):
        self.top_k = top_k or config.DEFAULT_TOP_K
        self.history: List[dict] = []          # [{"role": "user"/"assistant", "content": ...}, ...]
        self.last_results: List[dict] = []      # most recent search results, for follow-up questions

    def _remember(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def _search_intent(self, message: str) -> Optional[str]:
        m = re.search(
            r"(?:find|search for)\s+papers?\s+(?:about|on)\s+(.+)"
            r"|papers?\s+(?:about|on)\s+(.+)"
            r"|(?:find|search for)\s+(.+)",
            message, re.IGNORECASE,
        )
        if not m:
            return None
        topic = next(g for g in m.groups() if g)
        return topic.strip().rstrip("?.")

    def _wants_most_relevant(self, message: str) -> bool:
        return bool(re.search(r"most relevant|best (?:paper|match)|top paper", message, re.IGNORECASE))

    def _wants_comparison(self, message: str) -> bool:
        return bool(re.search(r"compare", message, re.IGNORECASE))

    def _wants_datasets(self, message: str) -> bool:
        return bool(re.search(r"datasets? (?:were|are|used|commonly)", message, re.IGNORECASE))

    def _wants_gaps(self, message: str) -> bool:
        return bool(re.search(r"research gaps?|underexplored|unexplored", message, re.IGNORECASE))

    def _wants_future_directions(self, message: str) -> bool:
        return bool(re.search(r"future (?:research|work|direction)", message, re.IGNORECASE))

    def _wants_lit_review(self, message: str) -> bool:
        return bool(re.search(r"literature review", message, re.IGNORECASE))

    def chat(self, message: str) -> dict:
        """
        Process one user turn. Returns:
            {"answer": str, "sources": [...], "structured": Optional[dict]}
        """
        self._remember("user", message)

        # 1. Explicit search intent
        query = self._search_intent(message)
        if query:
            results = semantic_search.search_papers(query, top_k=self.top_k)
            self.last_results = results
            if not results:
                answer = f"No papers were found matching '{query}'."
            else:
                lines = [f"Found {len(results)} paper(s) related to '{query}':"]
                for r in results[:5]:
                    lines.append(f"- {r['title']} ({r['year']}) -- similarity {r['similarity_score']}")
                answer = "\n".join(lines)
            self._remember("assistant", answer)
            return {"answer": answer, "sources": results, "structured": None}

        # 2. Ranking / "most relevant" over prior results
        if self._wants_most_relevant(message) and self.last_results:
            ranked = ranking.compute_scores(self.last_results)
            top = ranked[0]
            answer = (
                f"The most relevant paper is \"{top['title']}\" ({top['year']}), "
                f"with an overall score of {top['final_score']} "
                f"(semantic similarity {top['semantic_score']}, citation impact {top['citation_score']})."
            )
            self._remember("assistant", answer)
            return {"answer": answer, "sources": ranked[:1], "structured": {"ranked": ranked}}

        # 3. Comparison of top two results
        if self._wants_comparison(message) and len(self.last_results) >= 2:
            comparison = paper_comparison.compare_papers(self.last_results[0], self.last_results[1])
            answer = (
                f"Comparing \"{comparison['paper_a']['title']}\" and "
                f"\"{comparison['paper_b']['title']}\" -- see the structured comparison below."
            )
            self._remember("assistant", answer)
            return {"answer": answer, "sources": self.last_results[:2], "structured": comparison}

        # 4. Common datasets across last results
        if self._wants_datasets(message) and self.last_results:
            datasets = set()
            for p in self.last_results:
                summary = paper_comparison._summarize_paper(p)
                if summary["dataset"] != "Not explicitly stated in available metadata.":
                    datasets.update(d.strip() for d in summary["dataset"].split(","))
            answer = (
                f"Commonly referenced datasets across the retrieved papers: {', '.join(sorted(datasets))}."
                if datasets else
                "No specific datasets could be identified from the retrieved papers' abstracts."
            )
            self._remember("assistant", answer)
            return {"answer": answer, "sources": self.last_results, "structured": {"datasets": sorted(datasets)}}

        # 5. Research gaps
        if self._wants_gaps(message):
            papers = self.last_results or semantic_search.search_papers(message, top_k=self.top_k)
            gaps = research_gap.analyze_research_gaps(papers)
            answer = (
                f"Analyzed {gaps['num_papers_analyzed']} papers. "
                f"Found {len(gaps['underrepresented_topics'])} underrepresented topics and "
                f"{len(gaps['missing_combinations'])} potential unexplored topic combinations. "
                "See the structured breakdown below (observed evidence vs. hypotheses are labeled separately)."
            )
            self._remember("assistant", answer)
            return {"answer": answer, "sources": papers, "structured": gaps}

        # 6. Future research directions
        if self._wants_future_directions(message):
            papers = self.last_results or semantic_search.search_papers(message, top_k=self.top_k)
            gaps = research_gap.analyze_research_gaps(papers)
            suggestions = gaps["future_work_suggestions"]
            answer = (
                f"Found {len(suggestions)} paper(s) that explicitly discuss future work."
                if suggestions else
                "None of the retrieved papers explicitly discuss future research directions in their abstracts."
            )
            self._remember("assistant", answer)
            return {"answer": answer, "sources": papers, "structured": {"future_work": suggestions}}

        # 7. Literature review
        topic_match = re.search(r"literature review (?:on|about)\s+(.+)", message, re.IGNORECASE)
        if self._wants_lit_review(message):
            topic = topic_match.group(1).strip().rstrip("?.") if topic_match else message
            papers = semantic_search.search_papers(topic, top_k=self.top_k)
            review = literature_review.generate_literature_review(topic, papers)
            answer = f"Generated a literature review on '{topic}' organized into {len(review['themes'])} theme(s)."
            self._remember("assistant", answer)
            return {"answer": answer, "sources": papers, "structured": review}

        # 8. Fall through to full RAG pipeline
        result = rag.generate_answer(message, top_k=self.top_k)
        self.last_results = [
            {**s, "similarity_score": 0.0} for s in result["sources"]
        ] if result["sources"] else self.last_results
        self._remember("assistant", result["answer"])
        return {"answer": result["answer"], "sources": result["sources"], "structured": None}

    def reset(self) -> None:
        self.history = []
        self.last_results = []
