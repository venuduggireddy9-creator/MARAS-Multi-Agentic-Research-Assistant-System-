from retrieval.semantic_retriever import hybrid_ranking
from agents.query_domain_agent import detect_domains, paper_matches_domain


def rank_papers(query, papers, top_k=5):
    domains = detect_domains(query)
    ranked = hybrid_ranking(query, papers, top_k=len(papers) if papers else 0)
    boosted = []

    for p, score in ranked:
        text = (p.get("title", "") + " " + p.get("abstract", "")).strip()
        adjusted = score
        for d in domains:
            if paper_matches_domain(d, text):
                adjusted += 0.2
        boosted.append((p, adjusted))

    boosted.sort(key=lambda x: x[1], reverse=True)

    final = []
    for i, (p, s) in enumerate(boosted[:top_k]):
        p["score"] = round(s, 4)
        p["rank"] = i + 1
        final.append(p)

    return final
