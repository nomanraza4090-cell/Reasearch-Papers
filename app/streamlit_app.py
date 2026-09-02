"""
ResearchMind — Premium Streamlit Interface
===========================================

Run:
    streamlit run app/streamlit_app.py

This file focuses on UI/UX and page orchestration.
Existing business logic in src/ and app/components/ remains unchanged.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# STREAMLIT
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ResearchMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# PREMIUM CSS
# ---------------------------------------------------------------------------

def inject_premium_css():
    st.markdown(
        """
        <style>

        /* ================================================================
           GLOBAL
        ================================================================ */

        @import url(
            'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
        );

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(
                    circle at 5% 5%,
                    rgba(124, 58, 237, 0.20),
                    transparent 28%
                ),
                radial-gradient(
                    circle at 95% 10%,
                    rgba(6, 182, 212, 0.16),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 50% 100%,
                    rgba(236, 72, 153, 0.10),
                    transparent 35%
                ),
                #070b17;
            color: #f8fafc;
        }

        .main .block-container {
            max-width: 1450px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        /* ================================================================
           SIDEBAR
        ================================================================ */

        section[data-testid="stSidebar"] {
            background:
                linear-gradient(
                    180deg,
                    rgba(15, 23, 42, 0.98),
                    rgba(8, 12, 25, 0.98)
                );
            border-right: 1px solid rgba(148, 163, 184, 0.12);
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 1.5rem;
        }

        .sidebar-brand {
            padding: 10px 8px 24px 8px;
            text-align: center;
        }

        .sidebar-logo {
            width: 62px;
            height: 62px;
            margin: auto;
            border-radius: 20px;

            display: flex;
            align-items: center;
            justify-content: center;

            font-size: 32px;

            background:
                linear-gradient(
                    135deg,
                    #7c3aed,
                    #2563eb,
                    #06b6d4
                );

            box-shadow:
                0 12px 35px rgba(79, 70, 229, 0.35);
        }

        .sidebar-title {
            margin-top: 12px;
            font-size: 22px;
            font-weight: 800;
            color: white;
        }

        .sidebar-subtitle {
            font-size: 12px;
            color: #94a3b8;
            margin-top: 4px;
        }

        /* ================================================================
           HERO
        ================================================================ */

        .hero {
            position: relative;
            overflow: hidden;

            padding: 38px 42px;
            margin-bottom: 28px;

            border-radius: 28px;

            background:
                linear-gradient(
                    135deg,
                    rgba(124, 58, 237, 0.34),
                    rgba(37, 99, 235, 0.24),
                    rgba(6, 182, 212, 0.18)
                );

            border: 1px solid rgba(255,255,255,0.13);

            box-shadow:
                0 25px 80px rgba(0,0,0,0.28);
        }

        .hero::before {
            content: "";
            position: absolute;

            width: 240px;
            height: 240px;

            right: -80px;
            top: -100px;

            border-radius: 50%;

            background: rgba(255,255,255,0.08);
            filter: blur(5px);
        }

        .hero-badge {
            display: inline-block;

            padding: 7px 14px;

            border-radius: 999px;

            background: rgba(255,255,255,0.10);

            border: 1px solid rgba(255,255,255,0.12);

            color: #c4b5fd;

            font-size: 12px;
            font-weight: 700;

            margin-bottom: 14px;
        }

        .hero-title {
            font-size: clamp(34px, 5vw, 58px);
            line-height: 1.05;
            font-weight: 800;
            margin: 0;
            color: white;
        }

        .gradient-text {
            background:
                linear-gradient(
                    90deg,
                    #c084fc,
                    #60a5fa,
                    #22d3ee
                );

            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-description {
            max-width: 780px;

            margin-top: 16px;

            color: #cbd5e1;

            font-size: 16px;
            line-height: 1.7;
        }

        /* ================================================================
           SECTION HEADERS
        ================================================================ */

        .section-title {
            font-size: 25px;
            font-weight: 800;
            color: white;

            margin-top: 25px;
            margin-bottom: 6px;
        }

        .section-subtitle {
            color: #94a3b8;
            font-size: 14px;

            margin-bottom: 18px;
        }

        /* ================================================================
           METRIC CARDS
        ================================================================ */

        .metric-card {
            position: relative;
            overflow: hidden;

            min-height: 145px;

            padding: 22px;

            border-radius: 20px;

            background:
                linear-gradient(
                    145deg,
                    rgba(255,255,255,0.09),
                    rgba(255,255,255,0.035)
                );

            border: 1px solid rgba(255,255,255,0.10);

            box-shadow:
                0 12px 35px rgba(0,0,0,0.18);

            transition:
                transform 0.25s ease,
                border-color 0.25s ease;
        }

        .metric-card:hover {
            transform: translateY(-5px);
            border-color: rgba(139,92,246,0.45);
        }

        .metric-icon {
            font-size: 26px;
            margin-bottom: 10px;
        }

        .metric-value {
            font-size: 30px;
            font-weight: 800;
            color: white;
        }

        .metric-label {
            margin-top: 4px;
            color: #94a3b8;
            font-size: 13px;
        }

        /* ================================================================
           FEATURE CARDS
        ================================================================ */

        .feature-card {
            padding: 24px;

            min-height: 180px;

            border-radius: 22px;

            background:
                linear-gradient(
                    145deg,
                    rgba(30,41,59,0.72),
                    rgba(15,23,42,0.60)
                );

            border: 1px solid rgba(148,163,184,0.13);

            transition: all 0.25s ease;
        }

        .feature-card:hover {
            transform: translateY(-5px);

            border-color:
                rgba(96,165,250,0.45);

            box-shadow:
                0 18px 45px rgba(0,0,0,0.22);
        }

        .feature-icon {
            font-size: 30px;
            margin-bottom: 12px;
        }

        .feature-title {
            font-size: 17px;
            font-weight: 700;
            color: white;
        }

        .feature-text {
            margin-top: 8px;
            font-size: 13px;
            line-height: 1.6;
            color: #94a3b8;
        }

        /* ================================================================
           PAPER CARD
        ================================================================ */

        .paper-card {
            padding: 22px;

            margin: 12px 0;

            border-radius: 18px;

            background:
                rgba(15,23,42,0.70);

            border:
                1px solid rgba(148,163,184,0.12);

            transition: all 0.2s ease;
        }

        .paper-card:hover {
            border-color:
                rgba(124,58,237,0.45);

            transform: translateY(-2px);
        }

        .paper-title {
            font-size: 17px;
            font-weight: 700;
            color: white;
        }

        .paper-meta {
            margin-top: 8px;
            color: #94a3b8;
            font-size: 12px;
        }

        .paper-description {
            margin-top: 12px;
            color: #cbd5e1;
            font-size: 13px;
            line-height: 1.6;
        }

        /* ================================================================
           CHAT
        ================================================================ */

        .rm-chat-row {
            display: flex;
            margin: 12px 0;
        }

        .rm-chat-user {
            justify-content: flex-end;
        }

        .rm-chat-assistant {
            justify-content: flex-start;
        }

        .rm-bubble {
            max-width: 78%;

            padding: 13px 17px;

            border-radius: 18px;

            line-height: 1.6;

            font-size: 14px;
        }

        .rm-bubble-user {
            background:
                linear-gradient(
                    135deg,
                    #7c3aed,
                    #2563eb
                );

            color: white;

            border-bottom-right-radius: 5px;
        }

        .rm-bubble-assistant {
            background:
                rgba(30,41,59,0.85);

            color: #e2e8f0;

            border:
                1px solid rgba(148,163,184,0.12);

            border-bottom-left-radius: 5px;
        }

        /* ================================================================
           BUTTONS
        ================================================================ */

        .stButton > button {
            border-radius: 12px;

            border: 1px solid rgba(148,163,184,0.15);

            background:
                rgba(255,255,255,0.06);

            color: white;

            font-weight: 600;

            transition: all 0.2s ease;
        }

        .stButton > button:hover {
            border-color: #8b5cf6;

            background:
                rgba(124,58,237,0.18);

            transform: translateY(-1px);
        }

        /* ================================================================
           INPUTS
        ================================================================ */

        div[data-baseweb="input"],
        div[data-baseweb="textarea"],
        div[data-baseweb="select"] {
            border-radius: 12px !important;
        }

        /* ================================================================
           DATAFRAME
        ================================================================ */

        [data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(148,163,184,0.12);
        }

        /* ================================================================
           ALERTS
        ================================================================ */

        .stAlert {
            border-radius: 14px;
        }

        /* ================================================================
           FOOTER
        ================================================================ */

        .footer {
            margin-top: 60px;

            padding-top: 20px;

            text-align: center;

            border-top:
                1px solid rgba(148,163,184,0.10);

            color: #64748b;

            font-size: 12px;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# CSS FILE SUPPORT
# ---------------------------------------------------------------------------

def load_existing_css():
    """
    Load the project's existing CSS after the premium base theme.
    This preserves any component-specific styles already created.
    """

    css_path = PROJECT_ROOT / "app" / "assets" / "style.css"

    if css_path.exists():
        try:
            st.markdown(
                f"<style>{css_path.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True,
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------

def render_hero():

    st.markdown(
        """
        <div class="hero">

            <div class="hero-badge">
                ✨ AI-POWERED RESEARCH INTELLIGENCE
            </div>

            <h1 class="hero-title">
                Discover.
                <span class="gradient-text">Analyze.</span>
                Research.
            </h1>

            <p class="hero-description">
                ResearchMind transforms academic literature into actionable
                research intelligence using semantic search, paper ranking,
                research-gap analysis, literature reviews and grounded
                research conversations.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# HOME / LANDING PAGE
# ---------------------------------------------------------------------------

def render_home_page():

    render_hero()

    st.markdown(
        """
        <div class="section-title">
            🚀 Research Intelligence Workspace
        </div>

        <div class="section-subtitle">
            Explore your research collection using intelligent analysis tools.
        </div>
        """,
        unsafe_allow_html=True,
    )

    features = [
        (
            "🔎",
            "Semantic Search",
            "Find papers based on meaning and research context rather than simple keywords.",
        ),
        (
            "📊",
            "Research Analytics",
            "Explore publication trends, citation patterns and research topics.",
        ),
        (
            "🏆",
            "Paper Ranking",
            "Rank papers using semantic relevance, citations, recency and quality.",
        ),
        (
            "🔬",
            "Research Gaps",
            "Identify underexplored topics, limitations and potential research directions.",
        ),
        (
            "📚",
            "Literature Review",
            "Organize retrieved research into themes and structured evidence.",
        ),
        (
            "💬",
            "Research Chatbot",
            "Ask questions about your research collection using grounded retrieval.",
        ),
    ]

    cols = st.columns(3)

    for index, (icon, title, description) in enumerate(features):

        with cols[index % 3]:

            st.markdown(
                f"""
                <div class="feature-card">

                    <div class="feature-icon">
                        {icon}
                    </div>

                    <div class="feature-title">
                        {title}
                    </div>

                    <div class="feature-text">
                        {description}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-title">
            🧭 Recommended Workflow
        </div>

        <div class="section-subtitle">
            Follow the research pipeline from discovery to analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    workflow = st.columns(5)

    steps = [
        ("01", "Discover", "Search relevant papers"),
        ("02", "Retrieve", "Find semantic matches"),
        ("03", "Rank", "Prioritize strong papers"),
        ("04", "Analyze", "Identify themes & gaps"),
        ("05", "Research", "Generate insights"),
    ]

    for col, (number, title, description) in zip(workflow, steps):

        with col:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-icon">
                        {number}
                    </div>

                    <div class="feature-title">
                        {title}
                    </div>

                    <div class="feature-text">
                        {description}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# PIPELINE CHECK
# ---------------------------------------------------------------------------

def pipeline_ready() -> bool:

    missing = []

    if not config.FINAL_CSV_PATH.exists():
        missing.append(
            "Final dataset — run `02_Data_Preprocessing.ipynb`"
        )

    if not config.EMBEDDINGS_PATH.exists():
        missing.append(
            "Embeddings — run `04_Embedding_Generation.ipynb`"
        )

    if not config.FAISS_INDEX_PATH.exists():
        missing.append(
            "FAISS vector index — run `05_Vector_Database.ipynb`"
        )

    if missing:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-icon">⚠️</div>

                <div class="feature-title">
                    Research Pipeline Not Ready
                </div>

                <div class="feature-text">
                    Some required research artifacts have not been generated.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        for item in missing:
            st.warning(item)

        st.info(
            "Run the required notebooks in order, or execute `python run.py` "
            "to check the project readiness."
        )

        return False

    return True


# ---------------------------------------------------------------------------
# RESEARCH GAP PAGE
# ---------------------------------------------------------------------------

def render_research_gap_page():

    render_hero()

    st.markdown(
        """
        <div class="section-title">
            🔬 Research Gap Analysis
        </div>

        <div class="section-subtitle">
            Discover underexplored areas, recurring limitations and potential
            research directions from your retrieved literature.
        </div>
        """,
        unsafe_allow_html=True,
    )

    topic = st.text_input(
        "Research topic or question",
        placeholder="e.g. Large language models in clinical decision support",
        key="gap_topic",
    )

    top_k = st.slider(
        "Papers to analyze",
        min_value=5,
        max_value=30,
        value=15,
        key="gap_topk",
    )

    if not topic:

        st.info(
            "💡 Enter a research topic above to begin the analysis."
        )
        return

    with st.spinner("Analyzing research literature..."):

        papers = semantic_search.search_papers(
            topic,
            top_k=top_k,
        )

    if not papers:

        st.warning("No relevant papers were found.")
        return

    gaps = research_gap.analyze_research_gaps(papers)

    st.markdown(
        """
        <div class="metric-card">

            <div class="metric-icon">🧠</div>

            <div class="metric-value">
                Research Intelligence
            </div>

            <div class="metric-label">
                Evidence extracted from the retrieved literature
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📌 Frequently Studied Topics")

    st.dataframe(
        gaps["frequent_topics"],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 🧩 Underrepresented Topics")

    st.dataframe(
        gaps["underrepresented_topics"],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### 💡 Potential Unexplored Topic Combinations"
    )

    for combo in gaps["missing_combinations"]:

        st.markdown(
            f"""
            <div class="paper-card">
                <div class="paper-title">
                    💡 {html.escape(str(combo["combination"]))}
                </div>
                <div class="paper-meta">
                    Hypothesis — requires independent validation
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "### ⚠️ Methodological & Dataset Limitations"
    )

    for item in gaps["methodological_limitations"]:

        st.markdown(
            f"""
            <div class="paper-card">

                <div class="paper-title">
                    {html.escape(str(item["title"]))}
                </div>

                <div class="paper-description">
                    {html.escape(str(item["snippet"]))}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🔀 Contradictory Findings")

    if gaps["contradictory_findings"]:

        for item in gaps["contradictory_findings"]:

            st.markdown(
                f"""
                <div class="paper-card">

                    <div class="paper-title">
                        {html.escape(str(item["title"]))}
                    </div>

                    <div class="paper-description">
                        {html.escape(str(item["snippet"]))}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    else:

        st.info(
            "No explicit contradictions were detected in the retrieved abstracts."
        )

    st.markdown("### 🔭 Future Work Suggestions")

    for item in gaps["future_work_suggestions"]:

        st.markdown(
            f"""
            <div class="paper-card">

                <div class="paper-title">
                    {html.escape(str(item["title"]))}
                </div>

                <div class="paper-description">
                    {html.escape(str(item["suggestion"]))}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(gaps["disclaimer"])


# ---------------------------------------------------------------------------
# LITERATURE REVIEW
# ---------------------------------------------------------------------------

def render_literature_review_page():

    render_hero()

    st.markdown(
        """
        <div class="section-title">
            📚 Literature Review
        </div>

        <div class="section-subtitle">
            Transform retrieved papers into structured research themes.
        </div>
        """,
        unsafe_allow_html=True,
    )

    topic = st.text_input(
        "Research topic",
        placeholder="e.g. AI chatbots in patient communication",
        key="literature_topic",
    )

    top_k = st.slider(
        "Papers to include",
        5,
        30,
        15,
        key="litrev_topk",
    )

    if not topic:

        st.info("💡 Enter a topic to generate a literature review.")
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

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-icon">📖</div>

            <div class="metric-value">
                {review["num_papers"]} Papers
            </div>

            <div class="metric-label">
                Organized into {len(review["themes"])} research themes
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"### 📖 Literature Review: {html.escape(topic)}"
    )

    for theme in review["themes"]:

        with st.expander(
            f"🧩 {theme['theme'].title()} — {theme['num_papers']} papers",
            expanded=True,
        ):

            st.write(theme["summary"])

            for paper in theme["papers"]:

                st.markdown(
                    f"""
                    <div class="paper-card">

                        <div class="paper-title">
                            {html.escape(str(paper["title"]))}
                        </div>

                        <div class="paper-meta">
                            📅 {paper["year"]}
                            &nbsp; • &nbsp;
                            👥 {html.escape(str(paper["authors"]))}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("### 🔗 References")

    for reference in review["references"]:
        st.markdown(f"- {reference}")


# ---------------------------------------------------------------------------
# CHATBOT
# ---------------------------------------------------------------------------

def render_chatbot_page():

    render_hero()

    st.markdown(
        """
        <div class="section-title">
            💬 ResearchMind AI
        </div>

        <div class="section-subtitle">
            Ask questions about your research collection and explore retrieved
            evidence through a conversational interface.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "chatbot" not in st.session_state:
        st.session_state["chatbot"] = chatbot.ResearchChatbot()

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    bot = st.session_state["chatbot"]

    if not st.session_state["chat_messages"]:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-icon">🧠</div>

                <div class="feature-title">
                    Ask ResearchMind
                </div>

                <div class="feature-text">
                    Try questions such as:
                    <br><br>
                    • What are the major research gaps in RAG?
                    <br>
                    • Compare the main approaches used in these papers.
                    <br>
                    • What datasets are commonly used?
                    <br>
                    • Summarize the recent research trends.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    for msg in st.session_state["chat_messages"]:

        role = msg["role"]

        row_class = (
            "rm-chat-user"
            if role == "user"
            else "rm-chat-assistant"
        )

        bubble_class = (
            "rm-bubble-user"
            if role == "user"
            else "rm-bubble-assistant"
        )

        content = html.escape(
            str(msg["content"])
        ).replace("\n", "<br>")

        st.markdown(
            f"""
            <div class="rm-chat-row {row_class}">
                <div class="rm-bubble {bubble_class}">
                    {content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if msg.get("sources"):

            with st.expander(
                f"📚 Sources ({len(msg['sources'][:5])})"
            ):

                for source in msg["sources"][:5]:

                    title = html.escape(
                        str(source.get("title", ""))
                    )

                    year = html.escape(
                        str(source.get("year", "n.d."))
                    )

                    url = source.get("url", "")

                    if url:

                        st.markdown(
                            f"- **{title}** ({year}) — "
                            f"[Open Paper]({url})"
                        )

                    else:

                        st.markdown(
                            f"- **{title}** ({year})"
                        )

    user_input = st.chat_input(
        "Ask ResearchMind about your research..."
    )

    if user_input:

        st.session_state["chat_messages"].append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        with st.spinner("ResearchMind is analyzing your literature..."):

            response = bot.chat(user_input)

        st.session_state["chat_messages"].append(
            {
                "role": "assistant",
                "content": response["answer"],
                "sources": response.get("sources", []),
            }
        )

        st.rerun()

    if st.session_state["chat_messages"]:

        if st.button(
            "🗑️ Clear Conversation",
            use_container_width=False,
        ):

            bot.reset()

            st.session_state["chat_messages"] = []

            st.rerun()


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------

def render_footer():

    st.markdown(
        """
        <div class="footer">

            🧠 <b>ResearchMind</b>
            &nbsp; • &nbsp;
            AI Research Intelligence Platform
            &nbsp; • &nbsp;
            Version 1.0

            <br><br>

            Semantic Search · Paper Ranking · Research Gaps ·
            Literature Review · RAG

        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    inject_premium_css()
    load_existing_css()

    # ---------------------------------------------------------------
    # SIDEBAR
    # ---------------------------------------------------------------

    with st.sidebar:

        st.markdown(
            """
            <div class="sidebar-brand">

                <div class="sidebar-logo">
                    🧠
                </div>

                <div class="sidebar-title">
                    ResearchMind
                </div>

                <div class="sidebar-subtitle">
                    Research Intelligence Platform
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        page = st.radio(
            "WORKSPACE",
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
            label_visibility="collapsed",
        )

        st.markdown("---")

        st.markdown(
            """
            <div style="
                padding: 14px;
                border-radius: 14px;
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.08);
            ">

                <div style="
                    color:#94a3b8;
                    font-size:11px;
                    font-weight:700;
                ">
                    SYSTEM
                </div>

                <div style="
                    color:#22c55e;
                    font-size:13px;
                    margin-top:6px;
                ">
                    ● Research workspace online
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")

        st.caption(
            "ResearchMind v1.0"
        )

    # ---------------------------------------------------------------
    # PAGE ROUTING
    # ---------------------------------------------------------------

    if page == "🏠 Home":

        render_home_page()

    elif page == "📊 Dashboard":

        if pipeline_ready():
            render_dashboard_page()

    elif page == "🔎 Semantic Search":

        if pipeline_ready():
            render_search_page()

    elif page == "🏆 Paper Ranking":

        if not pipeline_ready():
            return

        render_hero()

        st.markdown(
            """
            <div class="section-title">
                🏆 Paper Ranking
            </div>

            <div class="section-subtitle">
                Prioritize research using semantic relevance, citations,
                recency and quality signals.
            </div>
            """,
            unsafe_allow_html=True,
        )

        results = st.session_state.get(
            "last_search_results"
        )

        if not results:

            st.info(
                "🔎 Run a semantic search first to generate ranking results."
            )

        else:

            ranked = ranking.compute_scores(results)

            ranking_rows = []

            for result in ranked:

                ranking_rows.append(
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
                ranking_rows,
                use_container_width=True,
                hide_index=True,
            )

    elif page == "📑 Paper Comparison":

        if pipeline_ready():
            render_comparison_page()

    elif page == "🔬 Research Gap Analysis":

        if pipeline_ready():
            render_research_gap_page()

    elif page == "📚 Literature Review":

        if pipeline_ready():
            render_literature_review_page()

    elif page == "💬 Research Chatbot":

        if pipeline_ready():
            render_chatbot_page()

    render_footer()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
