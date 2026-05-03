import hashlib
import html
import os
import sys
import time

import fitz
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.query_domain_agent import detect_domains, refine_query
from agents.ranking_agent import rank_papers
from agents.retrieval_filter_agent import retrieve_papers
from agents import writing_agent
from main import run_pipeline

improve_paragraph = writing_agent.improve_paragraph
live_analyze = writing_agent.live_analyze
get_llm_debug_status = getattr(
    writing_agent,
    "get_llm_debug_status",
    lambda: {
        "api_key_loaded": bool(os.getenv("GROQ_API_KEY")),
        "client_initialized": False,
        "last_error": "Debug helper unavailable in deployed writing_agent version.",
    },
)

st.set_page_config(
    layout="wide",
    page_title="MULTI_AGENTIC RESEARCH ASSISTANT SYSTEM",
    page_icon="🧠",
    initial_sidebar_state="collapsed",
)

with st.sidebar.expander("Debug: LLM health", expanded=False):
    llm_status = get_llm_debug_status()
    st.write("GROQ_API_KEY loaded:", llm_status["api_key_loaded"])
    st.write("Groq client initialized:", llm_status["client_initialized"])
    if llm_status["last_error"]:
        st.error(f"Last LLM error: {llm_status['last_error']}")
    else:
        st.caption("No recent LLM errors.")

st.markdown(
    """
<style>
    :root {
        --bg-0: #050816;
        --bg-1: #0d1226;
        --bg-2: #151f3d;
        --text-main: #eef3ff;
        --text-muted: #b8c0df;
        --primary: #7c3aed;
        --primary-2: #22d3ee;
        --success: #10b981;
        --warn: #f59e0b;
    }

    .stApp {
        background:
            radial-gradient(800px 300px at 8% -10%, rgba(34, 211, 238, 0.24), transparent 60%),
            radial-gradient(900px 380px at 92% -12%, rgba(124, 58, 237, 0.28), transparent 65%),
            linear-gradient(155deg, var(--bg-0), var(--bg-1) 45%, var(--bg-2) 100%);
        color: var(--text-main);
    }

    .main > div {
        max-width: 1380px;
        padding-top: 1.1rem;
    }

    .wa-block-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin: 0 0 8px 0;
        font-weight: 700;
    }

    .wa-editor-wrap {
        background: rgba(15, 23, 42, 0.45);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 14px;
        padding: 14px 16px 16px;
    }

    .wa-feed-stack {
        display: flex;
        flex-direction: column;
        gap: 14px;
        max-height: none;
    }

    .style-improve-card {
        background: rgba(124, 58, 237, 0.10);
        border: 1px solid rgba(124, 58, 237, 0.35);
        border-radius: 12px;
        padding: 12px 14px;
        font-size: 14px;
        line-height: 1.55;
        color: #e8e9ff;
    }

    .maras-hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.30), rgba(34, 211, 238, 0.20));
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 18px;
        padding: 22px 24px;
        margin: 4px 0 18px;
        box-shadow: 0 12px 38px rgba(0, 0, 0, 0.24);
        backdrop-filter: blur(12px);
    }

    .maras-hero::after {
        content: "";
        position: absolute;
        inset: -120px -40px auto auto;
        width: 260px;
        height: 260px;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.22), transparent 62%);
        animation: floatOrb 9s ease-in-out infinite;
        pointer-events: none;
    }

    @keyframes floatOrb {
        0% { transform: translateY(0px) rotate(0deg); opacity: 0.75; }
        50% { transform: translateY(16px) rotate(10deg); opacity: 1; }
        100% { transform: translateY(0px) rotate(0deg); opacity: 0.75; }
    }

    .maras-title {
        margin: 0;
        font-size: 2rem;
        line-height: 1.2;
        font-weight: 800;
        letter-spacing: 0.2px;
        color: #f7f9ff;
    }

    .maras-sub {
        margin-top: 8px;
        margin-bottom: 0;
        color: var(--text-muted);
        font-size: 0.99rem;
        max-width: 840px;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 12px 14px;
        margin-bottom: 10px;
        backdrop-filter: blur(8px);
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
    }

    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.28);
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
    }

    .live-badge {
        display: inline-block;
        background: linear-gradient(90deg, #0f7c4a, #16a34a);
        color: white;
        font-size: 11px;
        padding: 3px 10px;
        border-radius: 999px;
        font-weight: 700;
        letter-spacing: 0.4px;
        box-shadow: 0 0 0 rgba(16, 185, 129, 0.25);
        animation: pulseGlow 2.2s infinite;
    }

    @keyframes pulseGlow {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.42); }
        70% { box-shadow: 0 0 0 9px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    .suggestion-box {
        background: rgba(124, 58, 237, 0.12);
        border: 1px solid rgba(124, 58, 237, 0.35);
        border-left: 3px solid #8b5cf6;
        padding: 10px 14px;
        border-radius: 10px;
        margin: 8px 0;
        font-size: 14px;
    }

    .hint-box {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-left: 3px solid #10b981;
        padding: 10px 14px;
        border-radius: 10px;
        margin: 8px 0;
        font-size: 14px;
    }

    .warn-box {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.34);
        border-left: 3px solid #f59e0b;
        padding: 10px 14px;
        border-radius: 10px;
        margin: 8px 0;
        font-size: 14px;
    }

    .section-guide {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.35);
        padding: 10px 14px;
        border-radius: 10px;
        font-size: 13px;
        color: #cbd5e1;
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.13);
        border-radius: 12px;
        padding: 10px 12px;
    }

    .stTextInput > div > div > input,
    .stTextArea textarea,
    .stSelectbox [data-baseweb="select"] > div,
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: var(--text-main) !important;
        border-radius: 10px !important;
    }

    .stButton > button {
        border: 0 !important;
        background: linear-gradient(90deg, var(--primary), #6d28d9 40%, var(--primary-2)) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        box-shadow: 0 10px 20px rgba(124, 58, 237, 0.25);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 24px rgba(124, 58, 237, 0.30);
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="maras-hero">
        <h1 class="maras-title">MULTI-AGENTIC RESEARCH ASSISTANT SYSTEM</h1>
        <p class="maras-sub">
        Intelligent research analysis made simple.  
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

query = st.text_input("Enter Research Topic", placeholder="e.g., Efficient multimodal LLMs for healthcare diagnostics")
col_time, col_count = st.columns(2)

with col_time:
    time_period = st.selectbox("Time Period", ["All", "2025-present", "2024-present", "2023-present"])
with col_count:
    num_papers = st.slider("Number of Papers", 1, 5, 5)

st.markdown("---")


def _extract_uploaded_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = " ".join(doc[i].get_text() for i in range(min(10, len(doc))))
    text = " ".join(text.split())[:8000]
    return {
        "title": f"[Uploaded] {uploaded_file.name.replace('.pdf', '')}",
        "abstract": text,
        "pdf_url": None,
        "year": "Uploaded",
        "authors": ["User Uploaded"],
    }


if st.button("Retrieve Papers", use_container_width=True):
    if not query:
        st.warning("Please enter a research topic")
        st.stop()

    with st.spinner("Fetching and ranking papers from arXiv..."):
        try:
            start = time.time()
            retrieval_query = refine_query(query)
            domains = detect_domains(query)
            papers = retrieve_papers(retrieval_query, max_results=num_papers * 3, time_filter=time_period)
            ranked = rank_papers(query, papers, num_papers) if papers else []

            st.session_state["retrieved"] = ranked
            st.session_state["retrieval_time"] = round(time.time() - start, 2)
            st.session_state["retrieval_query"] = retrieval_query
            st.session_state["domains"] = domains
            st.session_state.pop("result", None)
            st.session_state.pop("user_papers", None)
            st.session_state.pop("_feedback", None)
            st.session_state.pop("_feedback_key", None)

            if ranked:
                st.success(f"Retrieved {len(ranked)} papers")
            else:
                st.warning("No papers found. Try simplifying the query.")
        except Exception as exc:
            st.error(f"Retrieval error: {exc}")
            st.stop()


if "retrieved" in st.session_state:
    st.subheader("Retrieved Papers")
    st.caption(f"Time: {st.session_state.get('retrieval_time', 0)} sec")
    st.caption(f"Query: {st.session_state.get('retrieval_query', '')}")
    st.caption(f"Domains: {', '.join(st.session_state.get('domains', []))}")

    for paper in st.session_state["retrieved"]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(f"### {paper.get('title', '')} ({paper.get('year', '')})")
        if paper.get("pdf_url"):
            st.markdown(f"[Open PDF]({paper['pdf_url']})")
        st.write("Score:", round(paper.get("score", 0), 4))
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Upload Your PDFs (Optional, Max 2)")
    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_upload",
    )

    if uploaded_files:
        if len(uploaded_files) > 2:
            st.warning("Only the first 2 PDFs will be used.")
            uploaded_files = uploaded_files[:2]

        user_papers = []
        for uploaded_file in uploaded_files:
            try:
                user_papers.append(_extract_uploaded_pdf(uploaded_file))
                st.success(f"Extracted text from {uploaded_file.name}")
            except Exception as exc:
                st.error(f"{uploaded_file.name}: {exc}")

        st.session_state["user_papers"] = user_papers
    else:
        st.session_state.pop("user_papers", None)

    if st.button("Run Analysis", use_container_width=True):
        with st.spinner("Running analysis pipeline..."):
            try:
                base_papers = list(st.session_state.get("retrieved", []))
                user_papers = st.session_state.get("user_papers", [])
                merged_papers = base_papers + user_papers

                start = time.time()
                result = run_pipeline(
                    query=query,
                    max_papers=num_papers,
                    time_filter=time_period,
                    papers=merged_papers,
                )

                st.session_state["result"] = result
                st.session_state["analysis_time"] = round(time.time() - start, 2)
                st.session_state.pop("_feedback", None)
                st.session_state.pop("_feedback_key", None)
                st.success(f"Analysis complete in {st.session_state['analysis_time']} sec")
            except Exception as exc:
                st.error(f"Analysis error: {exc}")
                st.stop()


if "result" in st.session_state:
    result = st.session_state["result"]
    analyses = result.get("analyses", [])
    insights = result.get("insights", "")
    gaps = result.get("gaps", "")
    recommendations = result.get("recommendations", "")

    tab_analysis, tab_writing = st.tabs(["Analysis & results", "Writing assistant"])

    with tab_analysis:
        st.subheader("Pipeline timing")
        col_a, col_b, col_c = st.columns(3)
        retrieval_time = st.session_state.get("retrieval_time", 0)
        analysis_time = st.session_state.get("analysis_time", 0)
        col_a.metric("Retrieval time", f"{retrieval_time}s")
        col_b.metric("Analysis time", f"{analysis_time}s")
        col_c.metric("Total pipeline time", f"{round(retrieval_time + analysis_time, 2)}s")
        st.markdown("---")

        st.subheader("Top relevant papers")
        for paper in result.get("papers", []):
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown(f"### {paper.get('title', '')} ({paper.get('year', '')})")
            if paper.get("pdf_url"):
                st.markdown(f"[Open PDF]({paper['pdf_url']})")
            st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Paper-wise analysis")
        for analysis in result.get("analyses", []):
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown(f"### {analysis.get('title', '')}")
            for field in ["problem", "method", "dataset", "performance", "application", "limitations"]:
                st.markdown(f"**{field.replace('_', ' ').title()}:** {analysis.get(field, 'Not specified')}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Cross-paper insights")
        st.markdown(result.get("insights", ""))

        st.subheader("Comparison table")
        st.dataframe(pd.DataFrame(result.get("comparison", [])), use_container_width=True)

        st.subheader("Similarity heatmap")
        similarity = result.get("similarity")
        if similarity is not None and len(similarity) > 1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(similarity, annot=True, cmap="YlGnBu", ax=ax)
            st.pyplot(fig)
            st.caption("High similarity indicates closely related paper content; low similarity indicates diverse approaches.")
        else:
            st.info("Not enough papers for similarity.")

        st.subheader("Research gaps")
        st.markdown(result.get("gaps", ""))

        st.subheader("Recommendations")
        st.markdown(result.get("recommendations", ""))

    with tab_writing:
        st.markdown(
            "<p style='margin:0 0 12px 0;'><span class='live-badge'>LIVE</span> "
            "<span style='color:#b8c0df;font-size:0.95rem;'>Feedback refreshes when you pause typing (10+ characters).</span></p>",
            unsafe_allow_html=True,
        )

        col_editor, col_feedback = st.columns([0.40, 0.60], gap="large")

        with col_editor:
            st.markdown("<div class='wa-editor-wrap'>", unsafe_allow_html=True)
            st.markdown('<p class="wa-block-title">Your draft</p>', unsafe_allow_html=True)
            section = st.selectbox(
                "Section",
                ["Introduction", "Related Work", "Methodology", "Conclusion"],
                key="section_select",
            )
            user_text = st.text_area(
                "Paragraph",
                height=320,
                placeholder="Write your research paragraph here…",
                key="writing_text",
            )
            improve_col, clear_col = st.columns(2)
            with improve_col:
                run_improve = st.button("Improve paragraph", use_container_width=True, key="btn_improve_para")
            with clear_col:
                if st.button("Clear text", use_container_width=True, key="btn_clear_text"):
                    st.session_state["writing_text"] = ""
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

            if run_improve and user_text and len(user_text.strip()) >= 10:
                with st.spinner("Rewriting with research context…"):
                    improved = improve_paragraph(
                        user_text=user_text,
                        section=section,
                        analyses=analyses,
                        insights=insights,
                        gaps=gaps,
                        recommendations=recommendations,
                    )
                st.markdown('<p class="wa-block-title">Improved paragraph</p>', unsafe_allow_html=True)
                st.markdown(
                    "<div class='style-improve-card'>"
                    f"{html.escape(improved)}"
                    "</div>",
                    unsafe_allow_html=True,
                )
                if st.button("Replace draft with improved version", key="btn_use_improved"):
                    st.session_state["writing_text"] = improved
                    st.rerun()

        with col_feedback:
            st.markdown('<p class="wa-block-title" style="margin-bottom:10px;">Assistant feedback</p>', unsafe_allow_html=True)
            st.markdown("<div class='wa-feed-stack'>", unsafe_allow_html=True)

            if user_text and len(user_text.strip()) >= 10:
                content_key = hashlib.md5(f"{user_text}||{section}".encode()).hexdigest()

                if st.session_state.get("_feedback_key") != content_key:
                    with st.spinner("Generating suggestions…"):
                        feedback = live_analyze(
                            user_text=user_text,
                            section=section,
                            analyses=analyses,
                            insights=insights,
                            gaps=gaps,
                            recommendations=recommendations,
                        )
                    st.session_state["_feedback"] = feedback
                    st.session_state["_feedback_key"] = content_key
                else:
                    feedback = st.session_state.get("_feedback", {})

                if feedback.get("section_guide"):
                    st.markdown('<p class="wa-block-title">Section guide</p>', unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='section-guide'>{html.escape(feedback['section_guide'])}</div>",
                        unsafe_allow_html=True,
                    )

                if feedback.get("suggestions"):
                    st.markdown('<p class="wa-block-title">Live suggestions</p>', unsafe_allow_html=True)
                    for suggestion in feedback["suggestions"]:
                        st.markdown(
                            f"<div class='suggestion-box'>{html.escape(suggestion)}</div>",
                            unsafe_allow_html=True,
                        )

                if feedback.get("context_hints"):
                    st.markdown('<p class="wa-block-title">From retrieved papers</p>', unsafe_allow_html=True)
                    for hint in feedback["context_hints"]:
                        st.markdown(f"<div class='hint-box'>{html.escape(hint)}</div>", unsafe_allow_html=True)

                if feedback.get("style_improved"):
                    st.markdown('<p class="wa-block-title">Style improvement</p>', unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='style-improve-card'>{html.escape(feedback['style_improved'])}</div>",
                        unsafe_allow_html=True,
                    )

                flow_col, warn_col = st.columns(2, gap="small")
                with flow_col:
                    if feedback.get("flow_issue"):
                        st.markdown('<p class="wa-block-title">Flow</p>', unsafe_allow_html=True)
                        st.markdown(
                            f"<div class='warn-box' style='font-size:13px;'>{html.escape(feedback['flow_issue'])}</div>",
                            unsafe_allow_html=True,
                        )
                with warn_col:
                    if feedback.get("redundancy"):
                        st.markdown('<p class="wa-block-title">Wording</p>', unsafe_allow_html=True)
                        st.markdown(
                            f"<div class='warn-box' style='font-size:13px;'>{html.escape(feedback['redundancy'])}</div>",
                            unsafe_allow_html=True,
                        )

                if feedback.get("gap_alignment"):
                    st.markdown('<p class="wa-block-title">Gap alignment</p>', unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='warn-box'>{html.escape(feedback['gap_alignment'])}</div>",
                        unsafe_allow_html=True,
                    )

                if feedback.get("autocomplete"):
                    st.markdown('<p class="wa-block-title">Autocomplete</p>', unsafe_allow_html=True)
                    st.code(feedback["autocomplete"], language=None)
                    if st.button("Accept completion", key="btn_accept_completion"):
                        st.session_state["writing_text"] = user_text.rstrip() + " " + feedback["autocomplete"]
                        st.rerun()

                if feedback.get("metric_suggestion"):
                    st.markdown('<p class="wa-block-title">Suggested metrics</p>', unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='glass-card'>{html.escape(feedback['metric_suggestion'])}</div>",
                        unsafe_allow_html=True,
                    )

                if feedback.get("experiment_suggestion"):
                    st.markdown('<p class="wa-block-title">Experiment idea</p>', unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='glass-card'>{html.escape(feedback['experiment_suggestion'])}</div>",
                        unsafe_allow_html=True,
                    )

                if feedback.get("novel_idea"):
                    with st.expander("Novel research idea"):
                        st.markdown(feedback["novel_idea"])

            else:
                st.info("Type at least 10 characters in the editor on the left to see structured feedback here.")

            st.markdown("</div>", unsafe_allow_html=True)
