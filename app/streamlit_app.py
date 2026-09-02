"""
app/streamlit_app.py
=====================
ResearchMind — AI-powered research assistant.

Run with:
    streamlit run app/streamlit_app.py

This file wires together the reusable src/ business logic and the
app/components/ Streamlit UI pieces. It intentionally contains only
navigation + light page glue; all real logic lives in src/.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `from src import ...` / `from app.components import ...` when launched
# directly via `streamlit run app/streamlit_app.py` from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src import config, semantic_search, ranking, research_gap, literature_review, chatbot

from app.components.analytics import render_dashboard_page
from app.components.search import render_search_page
from app.components.comparison import render_comparison_page
from app.components.papers import render_paper_card


st.set_page_config(page_title="ResearchMind", page_icon="📚", layout="wide")


def _load_css():
    css_path = PROJECT_ROOT / "app" / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def _check_pipeline_ready() -> bool:
    """Give a clear, actionable message instead of a raw traceback if the
    pipeline (dataset / embeddings / index) has not been built yet."""
    missing = []
    if not config.FINAL_CSV_PATH.exists():
        missing.append("Final dataset (run notebooks/02_Data_Preprocessing.ipynb)")
    if not config.EMBEDDINGS_PATH.exists():
        missing.append("Embeddings (run notebooks/04_Embedding_Generation.ipynb)")
    if not config.FAISS_INDEX_PATH.exists():
        missing.append("Vector index (run notebooks/05_Vector_Database.ipynb)")

    if missing:
        st.error(
            "ResearchMind is not fully set up yet. Missing:\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n\nSee README.md, section 'Notebook Execution Order', or run `python run.py` "
              "for a full readiness check."
        )
        return False
    return True


def render_research_gap_page():
    st.subheader("Research Gap Analysis")
    st.caption("Established evidence and potential (unverified) hypotheses are always labeled separately.")

    topic = st.text_input("Topic or research question", placeholder="e.g. large language models in clinical decision support")
    top_k = st.slider("Number of papers to analyze", 5, 30, 15)

    if not topic:
        st.info("Enter a topic to analyze research gaps.")
        return

    papers = semantic_search.search_papers(topic, top_k=top_k)
    if not papers:
        st.warning("No papers found for this topic.")
        return

    gaps = research_gap.analyze_research_gaps(papers)

    st.markdown(
        "```\nEstablished Research\n        ↓\nCommon Methods\n        ↓\n"
        "Known Limitations\n        ↓\nUnderexplored Areas\n        ↓\nPotential Research Gaps\n```"
    )

    st.markdown("### Frequently Studied Topics *(observed)*")
    st.dataframe(gaps["frequent_topics"], use_container_width=True, hide_index=True)

    st.markdown("### Underrepresented Topics *(observed)*")
    st.dataframe(gaps["underrepresented_topics"], use_container_width=True, hide_index=True)

    st.markdown("### Potential Unexplored Topic Combinations *(hypothesis — not established fact)*")
    for combo in gaps["missing_combinations"]:
        st.markdown(f"- {combo['combination']}")

    st.markdown("### Methodological / Dataset Limitations Mentioned *(observed)*")
    for item in gaps["methodological_limitations"]:
        st.markdown(f"- **{item['title']}**: {item['snippet']}")

    st.markdown("### Contradictory Findings *(observed)*")
    if gaps["contradictory_findings"]:
        for item in gaps["contradictory_findings"]:
            st.markdown(f"- **{item['title']}**: {item['snippet']}")
    else:
        st.caption("No explicit contradictions detected in the retrieved abstracts.")

    st.markdown("### Future-Work Suggestions *(observed)*")
    for item in gaps["future_work_suggestions"]:
        st.markdown(f"- **{item['title']}**: {item['suggestion']}")

    st.caption(gaps["disclaimer"])


def render_literature_review_page():
    st.subheader("Literature Review")
    topic = st.text_input("Enter a topic for the literature review", placeholder="e.g. AI chatbots in patient communication")
    top_k = st.slider("Number of papers to include", 5, 30, 15, key="litrev_topk")

    if not topic:
        st.info("Enter a topic to generate a literature review.")
        return

    papers = semantic_search.search_papers(topic, top_k=top_k)
    review = literature_review.generate_literature_review(topic, papers)

    if review["num_papers"] == 0:
        st.warning(review.get("note", "No papers found."))
        return

    st.markdown(f"## Literature Review: {topic}")
    st.caption(f"Organized from {review['num_papers']} retrieved papers into {len(review['themes'])} theme(s).")

    for theme in review["themes"]:
        with st.expander(f"**{theme['theme'].title()}** ({theme['num_papers']} papers)", expanded=True):
            st.write(theme["summary"])
            for p in theme["papers"]:
                st.markdown(f"- *{p['title']}* ({p['year']}) — {p['authors']}")

    st.markdown("### References")
    for ref in review["references"]:
        st.markdown(f"- {ref}")


def render_chatbot_page():
    st.subheader("Research Chatbot")
    st.caption("Ask about papers, comparisons, datasets, research gaps, or request a literature review.")

    if "chatbot" not in st.session_state:
        st.session_state["chatbot"] = chatbot.ResearchChatbot()
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    bot = st.session_state["chatbot"]

    for msg in st.session_state["chat_messages"]:
        row_class = "rm-chat-user" if msg["role"] == "user" else "rm-chat-assistant"
        bubble_class = "rm-bubble-user" if msg["role"] == "user" else "rm-bubble-assistant"
        st.markdown(
            f'<div class="rm-chat-row {row_class}"><div class="rm-bubble {bubble_class}">{msg["content"]}</div></div>',
            unsafe_allow_html=True,
        )
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"][:5]:
                    st.markdown(f"- **{s.get('title','')}** ({s.get('year','n.d.')}) — [{s.get('url','')}]({s.get('url','')})")

    user_input = st.chat_input("Ask ResearchMind about research papers...")
    if user_input:
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            response = bot.chat(user_input)
        st.session_state["chat_messages"].append({
            "role": "assistant", "content": response["answer"], "sources": response.get("sources", []),
        })
        st.rerun()

    if st.button("Clear conversation"):
        bot.reset()
        st.session_state["chat_messages"] = []
        st.rerun()


def main():
    _load_css()
    st.title("📚 ResearchMind")
    st.caption("AI-powered research paper discovery and analysis")

    page = st.sidebar.radio(
        "Navigate",
        ["Dashboard", "Semantic Search", "Paper Ranking", "Paper Comparison",
         "Research Gap Analysis", "Literature Review", "Research Chatbot"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("ResearchMind v1.0 — modular AI research assistant")

    if page == "Dashboard":
        render_dashboard_page()
        return

    if not _check_pipeline_ready():
        return

    if page == "Semantic Search":
        render_search_page()
    elif page == "Paper Ranking":
        st.subheader("Paper Ranking")
        st.caption("Ranks your last search results using the configurable weighted-scoring strategy.")
        results = st.session_state.get("last_search_results")
        if not results:
            st.info("Run a search on the Semantic Search page first.")
        else:
            ranked = ranking.compute_scores(results)
            st.dataframe(
                [{"Rank": r["rank"], "Paper": r["title"], "Relevance": r.get("semantic_score", r.get("similarity_score")),
                  "Citations": r.get("citation_count", 0), "Year": r.get("year"), "Overall Score": r["final_score"]}
                 for r in ranked],
                use_container_width=True, hide_index=True,
            )
    elif page == "Paper Comparison":
        render_comparison_page()
    elif page == "Research Gap Analysis":
        render_research_gap_page()
    elif page == "Literature Review":
        render_literature_review_page()
    elif page == "Research Chatbot":
        render_chatbot_page()


if __name__ == "__main__":
    main()
