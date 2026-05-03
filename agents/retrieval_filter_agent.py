import time

from retrieval.arxiv_api import fetch_arxiv_papers as search_arxiv
from retrieval.semantic_retriever import hybrid_ranking
from agents.query_domain_agent import detect_domains, generate_subqueries, paper_matches_domain

TITLE_FILTER_KEYWORDS = ["survey", "review", "overview"]
ABSTRACT_FILTER_PHRASES = ["this survey", "this review"]


def _is_survey_or_review(paper):
    title = paper.get("title", "").lower()
    abstract = paper.get("abstract", "").lower()
    return any(k in title for k in TITLE_FILTER_KEYWORDS) or any(
        p in abstract for p in ABSTRACT_FILTER_PHRASES
    )


def _enforce_domain(query, paper):
    domains = detect_domains(query)
    if not domains:
        return True

    text = paper.get("title", "") + " " + paper.get("abstract", "")
    score = sum(1 for d in domains if paper_matches_domain(d, text))
    return score >= 1


def retrieve_papers(query, max_results=5, time_filter="All"):
    start_time = time.time()
    try:
        subqueries = generate_subqueries(query, max_queries=3)
        print("🔍 Subqueries:", subqueries)

        fetch_per = max(max_results * 3 // max(len(subqueries), 1), max_results + 2)
        all_papers = []

        for sq in subqueries:
            results = search_arxiv(sq, fetch_per, time_filter)
            all_papers.extend(results)

        if not all_papers:
            print("⚠️ fallback retrieval")
            all_papers = search_arxiv(query.strip()[:120] or "machine learning", max_results * 3, time_filter)

        seen = set()
        papers = []
        for p in all_papers:
            title = p.get("title", "")
            if title and title not in seen:
                seen.add(title)
                papers.append(p)

        papers = [p for p in papers if not _is_survey_or_review(p)]
        if not papers:
            return []

        ranked = hybrid_ranking(query, papers, top_k=min(len(papers), max_results * 6))
        ranked_papers = [p for p, _ in ranked]

        filtered = [p for p in ranked_papers if _enforce_domain(query, p)]
        if len(filtered) < max_results:
            print("⚠️ Relaxing domain filter")
            filtered = ranked_papers[: max(len(ranked_papers), max_results)]

        if time_filter != "All":
            try:
                year_limit = int(time_filter.split("-")[0])
                temp = []
                for p in filtered:
                    y = p.get("year", 0)
                    try:
                        yi = int(y) if y not in (None, "", "Uploaded") else 0
                    except (TypeError, ValueError):
                        yi = 0
                    if yi >= year_limit:
                        temp.append(p)
                if temp:
                    filtered = temp
            except ValueError:
                pass

        final = filtered[:max_results]
        for i, p in enumerate(final):
            p["rank"] = i + 1

        elapsed = round(time.time() - start_time, 2)
        print(f"⏱️ Retrieval completed in {elapsed}s")
        return final

    except Exception as e:
        print("❌ Retrieval Error:", e)
        return []


def filter_papers(papers, domain, threshold=1):
    filtered = []
    for p in papers:
        text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
        score = sum(1 for w in domain.split() if w in text)
        if score >= threshold:
            filtered.append(p)
    return filtered
