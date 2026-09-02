"""
ResearchMind - Streamlit Application
====================================

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------

import streamlit as st

from src import (
    config,
    semantic_search,
    ranking,
    research_gap,
    literature_review,
    chatbot,
)

from app.components.analytics import render_dashboard_page
from app.components.search import render_search_page
from app.components.comparison import render_comparison_page


# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="ResearchMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# PREMIUM THEME
# ---------------------------------------------------------------------

def load_theme():

    st.markdown(
        """
        <style>

        /* Main background */
        .stApp {
            background: linear-gradient(
                135deg,
                #0f172a 0%,
                #111827 45%,
                #172554 100%
            );
        }

        /* Main container */
        .block-container {
            max-width: 1400px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #0b1120;
        }

        /* Sidebar text */
        section[data-testid="stSidebar"] * {
            color: #e2e8f0;
        }

        /* Main title */
        .main-title {
            font-size: 48px;
            font-weight: 800;
            margin-bottom: 5px;
            background: linear-gradient(
                90deg,
                #a78bfa,
                #60a5fa,
                #22d3ee
            );
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .main-subtitle {
            font-size: 17px;
            color: #94a3b8;
            margin-bottom: 30px;
        }

        /* Cards */
        .card {
            padding: 24px;
            border-radius: 18px;
            background: rgba(30, 41, 59, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.15);
            margin-bottom: 18px;
        }

        .card:hover {
            border-color: rgba(129, 140, 248, 0.5);
        }

        .card-title {
            font-size: 20px;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 8px;
        }

        .card-text {
            font-size: 14px;
            color: #94a3b8;
            line-height: 1.6;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.15);
            padding: 18px;
            border-radius: 16px;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
        }

        /* Inputs */
        input {
            border-radius: 10px !important;
        }

        /* Dataframes */
        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
        }

        /* Footer */
        .footer {
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            color: #64748b;
            font-size: 12px;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------

def render_header():

    st.markdown(
        '<div class="main-title">🧠 ResearchMind</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="main-subtitle">
            AI-powered research paper discovery, analysis and intelligence
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------

def render_home():

    render_header()

    st.markdown(
        """
        <div class="card">

        <div class="card-title">
        🚀 Research Intelligence Workspace
        </div>

        <div class="card-text">
        ResearchMind helps researchers discover relevant academic papers,
        perform semantic search, rank research, identify research gaps,
        generate literature reviews and interact with research through AI.
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("✨ Explore ResearchMind")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="card">

            <div class="card-title">
            🔎 Semantic Search
            </div>

            <div class="card-text">
            Search papers by meaning and context instead of relying only
            on exact keywords.
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="card">

            <div class="card-title">
            🏆 Paper Ranking
            </div>

            <div class="card-text">
            Rank papers using semantic relevance, citations,
            recency and quality signals.
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            """
            <div class="card">

            <div class="card-title">
            🔬 Research Gaps
            </div>

            <div class="card-text">
            Analyze literature for limitations, underrepresented
            topics and potential research directions.
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    col4, col5, col6 = st.columns(3)

    with col4:

        st.markdown(
            """
            <div class="card">

            <div class="card-title">
            📚 Literature Review
            </div>

            <div class="card-text">
            Organize retrieved papers into research themes and
            structured literature reviews.
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:

        st.markdown(
            """
            <div class="card">

            <div class="card-title">
            📑 Paper Comparison
            </div>

            <div class="card-text">
            Compare research papers and examine their differences,
            methods and findings.
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col6:

        st.markdown(
            """
            <div class="card">

            <div class="card-title">
            💬 Research Chatbot
            </div>

            <div class="card-text">
            Ask questions about your research collection through
            a retrieval-based AI assistant.
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("🧭 Research Workflow")

    workflow = st.columns(5)

    workflow_data = [
        ("01", "Discover"),
        ("02", "Search"),
        ("03", "Rank"),
        ("04", "Analyze"),
        ("05", "Research"),
    ]

    for column, (number, name) in zip(workflow, workflow_data):

        with column:

            st.metric(
                label=name,
                value=number,
            )


# ---------------------------------------------------------------------
# PIPELINE CHECK
# ---------------------------------------------------------------------

def pipeline_ready():

    missing = []

    if not config.FINAL_CSV_PATH.exists():
        missing.append(
            "data/processed/research_papers_final.csv"
        )

    if not config.EMBEDDINGS_PATH.exists():
        missing.append(
            "data/embeddings/paper_embeddings.npy"
        )

    if not config.FAISS_INDEX_PATH.exists():
        missing.append(
            "vector_db/faiss_index/index.faiss"
        )

    if missing:

        st.error("ResearchMind pipeline is not ready.")

        st.warning("Missing required files:")

        for file in missing:
            st.code(file)

        st.info(
            "Make sure these files are included in your deployed repository."
        )

        return False

    return True


# ---------------------------------------------------------------------
# RESEARCH GAP
# ---------------------------------------------------------------------

def render_research_gap():

    render_header()

    st.title("🔬 Research Gap Analysis")

    st.caption(
        "Identify underrepresented topics, limitations and potential research directions."
    )

    topic = st.text_input(
        "Research topic",
        placeholder="Example: Large Language Models in healthcare",
    )

    top_k = st.slider(
        "Number of papers",
        5,
        30,
        15,
    )

    if not topic:

        st.info("Enter a research topic to begin.")

        return

    with st.spinner("Analyzing research literature..."):

        papers = semantic_search.search_papers(
            topic,
            top_k=top_k,
        )

    if not papers:

        st.warning("No papers found.")

        return

    gaps = research_gap.analyze_research_gaps(
        papers
    )

    st.subheader("📌 Frequently Studied Topics")

    st.dataframe(
        gaps["frequent_topics"],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("🧩 Underrepresented Topics")

    st.dataframe(
        gaps["underrepresented_topics"],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("💡 Potential Topic Combinations")

    for item in gaps["missing_combinations"]:

        st.info(
            item["combination"]
        )

    st.subheader("⚠️ Methodological Limitations")

    for item in gaps["methodological_limitations"]:

        with st.expander(
            item["title"]
        ):

            st.write(
                item["snippet"]
            )

    st.subheader("🔭 Future Work")

    for item in gaps["future_work_suggestions"]:

        with st.expander(
            item["title"]
        ):

            st.write(
                item["suggestion"]
            )

    st.caption(
        gaps["disclaimer"]
    )


# ---------------------------------------------------------------------
# LITERATURE REVIEW
# ---------------------------------------------------------------------

def render_literature():

    render_header()

    st.title("📚 Literature Review")

    st.caption(
        "Generate a structured review from retrieved research papers."
    )

    topic = st.text_input(
        "Research topic",
        placeholder="Example: AI chatbots in healthcare",
    )

    top_k = st.slider(
        "Papers to include",
        5,
        30,
        15,
    )

    if not topic:

        st.info(
            "Enter a topic to generate a literature review."
        )

        return

    with st.spinner("Building literature review..."):

        papers = semantic_search.search_papers(
            topic,
            top_k=top_k,
        )

        review = literature_review.generate_literature_review(
            topic,
            papers,
        )

    if review["num_papers"] == 0:

        st.warning(
            review.get(
                "note",
                "No papers found.",
            )
        )

        return

    st.success(
        f"Analyzed {review['num_papers']} papers "
        f"across {len(review['themes'])} themes."
    )

    for theme in review["themes"]:

        with st.expander(
            f"🧩 {theme['theme'].title()} "
            f"({theme['num_papers']} papers)",
            expanded=True,
        ):

            st.write(
                theme["summary"]
            )

            for paper in theme["papers"]:

                st.markdown(
                    f"**{paper['title']}**  \n"
                    f"📅 {paper['year']} · "
                    f"👥 {paper['authors']}"
                )

    st.subheader("🔗 References")

    for reference in review["references"]:

        st.markdown(
            f"- {reference}"
        )


# ---------------------------------------------------------------------
# CHATBOT
# ---------------------------------------------------------------------

def render_chatbot():

    render_header()

    st.title("💬 ResearchMind AI")

    st.caption(
        "Ask questions about your research papers."
    )

    if "chatbot" not in st.session_state:

        st.session_state.chatbot = (
            chatbot.ResearchChatbot()
        )

    if "chat_messages" not in st.session_state:

        st.session_state.chat_messages = []

    bot = st.session_state.chatbot

    for message in st.session_state.chat_messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            if message.get("sources"):

                with st.expander(
                    "📚 Sources"
                ):

                    for source in message["sources"][:5]:

                        title = source.get(
                            "title",
                            "Untitled paper",
                        )

                        year = source.get(
                            "year",
                            "n.d.",
                        )

                        url = source.get(
                            "url"
                        )

                        if url:

                            st.markdown(
                                f"- **{title}** "
                                f"({year}) — "
                                f"[Open paper]({url})"
                            )

                        else:

                            st.markdown(
                                f"- **{title}** ({year})"
                            )

    user_input = st.chat_input(
        "Ask ResearchMind..."
    )

    if user_input:

        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        with st.chat_message("user"):

            st.markdown(
                user_input
            )

        with st.chat_message("assistant"):

            with st.spinner(
                "ResearchMind is thinking..."
            ):

                response = bot.chat(
                    user_input
                )

            st.markdown(
                response["answer"]
            )

            if response.get("sources"):

                with st.expander(
                    "📚 Sources"
                ):

                    for source in response["sources"][:5]:

                        title = source.get(
                            "title",
                            "Untitled",
                        )

                        year = source.get(
                            "year",
                            "n.d.",
                        )

                        url = source.get(
                            "url"
                        )

                        if url:

                            st.markdown(
                                f"- **{title}** "
                                f"({year}) — "
                                f"[Open paper]({url})"
                            )

                        else:

                            st.markdown(
                                f"- **{title}** ({year})"
                            )

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": response["answer"],
                "sources": response.get(
                    "sources",
                    [],
                ),
            }
        )

        st.rerun()

    if st.session_state.chat_messages:

        if st.button(
            "🗑️ Clear Conversation"
        ):

            bot.reset()

            st.session_state.chat_messages = []

            st.rerun()


# ---------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------

def footer():

    st.markdown(
        """
        <div class="footer">
        🧠 ResearchMind · AI Research Intelligence Platform · v1.0
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    load_theme()

    # Sidebar

    with st.sidebar:

        st.markdown(
            "# 🧠 ResearchMind"
        )

        st.caption(
            "AI Research Intelligence"
        )

        st.divider()

        page = st.radio(
            "Navigation",
            [
                "🏠 Home",
                "📊 Dashboard",
                "🔎 Semantic Search",
                "🏆 Paper Ranking",
                "📑 Paper Comparison",
                "🔬 Research Gap Analysis",
                "📚 Literature Review",
                "💬 Research Chatbot",
            ],
        )

        st.divider()

        st.caption(
            "ResearchMind v1.0"
        )

    # Pages

    if page == "🏠 Home":

        render_home()

    elif page == "📊 Dashboard":

        if pipeline_ready():

            render_header()

            render_dashboard_page()

    elif page == "🔎 Semantic Search":

        if pipeline_ready():

            render_header()

            render_search_page()

    elif page == "🏆 Paper Ranking":

        if not pipeline_ready():

            return

        render_header()

        st.title("🏆 Paper Ranking")

        results = st.session_state.get(
            "last_search_results"
        )

        if not results:

            st.info(
                "Run a Semantic Search first."
            )

        else:

            ranked = ranking.compute_scores(
                results
            )

            rows = []

            for result in ranked:

                rows.append(
                    {
                        "Rank": result["rank"],
                        "Paper": result["title"],
                        "Relevance": result.get(
                            "semantic_score",
                            result.get(
                                "similarity_score"
                            ),
                        ),
                        "Citations": result.get(
                            "citation_count",
                            0,
                        ),
                        "Year": result.get(
                            "year"
                        ),
                        "Overall Score": result[
                            "final_score"
                        ],
                    }
                )

            st.dataframe(
                rows,
                use_container_width=True,
                hide_index=True,
            )

    elif page == "📑 Paper Comparison":

        if pipeline_ready():

            render_header()

            render_comparison_page()

    elif page == "🔬 Research Gap Analysis":

        if pipeline_ready():

            render_research_gap()

    elif page == "📚 Literature Review":

        if pipeline_ready():

            render_literature()

    elif page == "💬 Research Chatbot":

        if pipeline_ready():

            render_chatbot()

    footer()


# ---------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
