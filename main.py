import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from agents.analysis_reasoning_agent import analyze_multiple
from agents.query_domain_agent import detect_domains, refine_query
from agents.ranking_agent import rank_papers
from agents.retrieval_filter_agent import retrieve_papers
from agents.synthesis_insight_agent import generate_insights
from retrieval.semantic_retriever import encode_texts


def compute_similarity(papers):
    if not papers:
        return np.array([])
    if len(papers) == 1:
        return np.array([[1.0]])

    try:
        texts = [
            f"{paper.get('title', '')} {paper.get('abstract', '')}".strip()
            for paper in papers
        ]
        embeddings = encode_texts(texts)
        if embeddings is None:
            return np.eye(len(papers))
        return np.round(cosine_similarity(embeddings), 2)
    except Exception as exc:
        print("Similarity error:", exc)
        return np.eye(len(papers))


def _normalize_paper(paper):
    normalized = dict(paper)
    normalized.setdefault("title", "Untitled")
    normalized.setdefault("abstract", "")
    normalized.setdefault("year", "")
    normalized.setdefault("authors", [])
    normalized.setdefault("pdf_url", None)
    return normalized


def _is_uploaded_paper(paper):
    title = paper.get("title", "")
    authors = paper.get("authors", [])
    return title.startswith("[Uploaded]") or "User Uploaded" in authors


def _collect_candidate_papers(query, max_papers, time_filter):
    """
    Single broad retrieval pass (subqueries capped inside retrieve_papers) for lower latency.
    """
    domains = detect_domains(query)
    domain_query = f"{query} {' '.join(domains)}".strip()
    fetch_n = max(max_papers * 5, max_papers + 5)

    seen_titles = set()
    merged = []

    for candidate in dict.fromkeys([query, domain_query] if domain_query != query else [query]):
        papers = retrieve_papers(candidate, max_results=fetch_n, time_filter=time_filter)
        for paper in papers:
            title = paper.get("title", "")
            if title and title not in seen_titles:
                seen_titles.add(title)
                merged.append(_normalize_paper(paper))

    return merged


def _agentic_iteration_loop(query, focus, papers, max_papers, rounds=1):
    ranking_context = f"{query} {focus}".strip()
    ranked = papers
    analyses = []
    insights = ""
    comparison = []
    gaps = ""
    recommendations = ""
    for _ in range(rounds):
        ranked = rank_papers(ranking_context, ranked, max_papers) if ranked else []
        analyses = analyze_multiple(ranked)
        comparison, insights, gaps, recommendations = generate_insights(analyses)

        if gaps and gaps.strip():
            ranking_context = f"{query} {focus} {gaps}".strip()

        for paper, analysis in zip(ranked, analyses):
            paper["fields"] = analysis
        for paper in ranked[len(analyses):]:
            paper["fields"] = {}

    return ranked, analyses, comparison, insights, gaps, recommendations


def run_pipeline(query, max_papers=5, time_filter="All", focus="general", papers=None):
    refined_query = refine_query(query)

    if papers is None:
        candidate_papers = _collect_candidate_papers(
            refined_query,
            max_papers=max_papers,
            time_filter=time_filter,
        )
    else:
        candidate_papers = [_normalize_paper(paper) for paper in papers]

    if not candidate_papers:
        return {
            "papers": [],
            "analyses": [],
            "insights": "",
            "comparison": [],
            "gaps": "",
            "recommendations": "",
            "similarity": np.array([]),
        }

    uploaded_papers = [paper for paper in candidate_papers if _is_uploaded_paper(paper)]
    retrieved_papers = [paper for paper in candidate_papers if not _is_uploaded_paper(paper)]

    if uploaded_papers:
        ranked, analyses, comparison, insights, gaps, recommendations = _agentic_iteration_loop(
            query=refined_query,
            focus=focus,
            papers=retrieved_papers,
            max_papers=max_papers,
            rounds=1,
        )
        for uploaded in uploaded_papers:
            uploaded["score"] = uploaded.get("score", 1.0)
            ranked.append(uploaded)
        for index, paper in enumerate(ranked, start=1):
            paper["rank"] = index
        analyses = analyze_multiple(ranked)
        comparison, insights, gaps, recommendations = generate_insights(analyses)
        for index, paper in enumerate(ranked):
            paper["fields"] = analyses[index] if index < len(analyses) else {}
    else:
        ranked, analyses, comparison, insights, gaps, recommendations = _agentic_iteration_loop(
            query=refined_query,
            focus=focus,
            papers=candidate_papers,
            max_papers=max_papers,
            rounds=1,
        )

    similarity = compute_similarity(ranked)

    return {
        "papers": ranked,
        "analyses": analyses,
        "insights": insights,
        "comparison": comparison,
        "gaps": gaps,
        "recommendations": recommendations,
        "similarity": similarity,
    }
