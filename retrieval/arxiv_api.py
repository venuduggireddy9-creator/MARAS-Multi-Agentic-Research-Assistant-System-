import arxiv
from datetime import datetime

# CS / ML acronyms: keep even when short (extract_keywords uses len >= 2)
CS_ACRONYMS = {
    "nlp", "gan", "gans", "rl", "cnn", "rnn", "lstm", "gru", "gnn", "gnns",
    "cv", "vae", "bert", "gpt", "llm", "llms", "mlp", "svm", "knn",
    "pde", "ode", "gpu", "tpu", "fl", "fls", "xai", "cvpr", "iclr",
    "neurips", "icml", "acl", "emnlp", "sigkdd", "www", "osdi", "nsdi",
}


def extract_keywords(query):
    words = query.lower().split()
    stopwords = {
        "the", "and", "for", "with", "using", "based", "systems", "models",
        "approach", "that", "this", "from", "into", "over", "are", "was",
    }
    keywords = []
    for w in words:
        w_clean = w.strip(".,;:()[]\"'").lower()
        if not w_clean or w_clean in stopwords:
            continue
        if w_clean in CS_ACRONYMS or len(w_clean) >= 2:
            keywords.append(w_clean)
    seen = set()
    ordered = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            ordered.append(k)
    return ordered[:8]


def build_arxiv_query(query):
    """
    Softer than all-AND: OR across terms for recall, optional AND on top anchors.
    """
    keywords = extract_keywords(query)
    if not keywords:
        return f'all:"{query.strip()[:200]}"'

    fields = [f'(ti:"{k}" OR abs:"{k}")' for k in keywords[:6]]
    if len(fields) <= 2:
        return " OR ".join(fields)

    core = " AND ".join(fields[:2])
    tail = " OR ".join(fields[2:])
    return f"({core}) OR ({tail})"


def min_year_from_time_filter(time_filter):
    if not time_filter or time_filter == "All":
        return None
    try:
        return int(str(time_filter).split("-")[0])
    except ValueError:
        return None


def fetch_arxiv_papers(query, max_papers=5, time_filter="All"):
    search_query = build_arxiv_query(query)
    min_year = min_year_from_time_filter(time_filter)

    search = arxiv.Search(
        query=search_query,
        max_results=max(max_papers * 4, 25),
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client()
    papers = []

    for paper in client.results(search):
        try:
            if min_year and paper.published.year < min_year:
                continue
            arxiv_id = paper.entry_id.split("/")[-1]
            papers.append({
                "title": paper.title,
                "abstract": paper.summary,
                "year": paper.published.year,
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            })
            if len(papers) >= max_papers:
                break
        except Exception:
            continue

    if not papers:
        try:
            fallback = arxiv.Search(
                query=f"all:{query.strip()[:180]}",
                max_results=max_papers,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            for paper in client.results(fallback):
                arxiv_id = paper.entry_id.split("/")[-1]
                papers.append({
                    "title": paper.title,
                    "abstract": paper.summary,
                    "year": paper.published.year,
                    "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
                })
                if len(papers) >= max_papers:
                    break
        except Exception:
            pass

    return papers


def get_publication_trend(query, start_year=2020):
    client = arxiv.Client()
    current_year = datetime.now().year
    search = arxiv.Search(
        query=f"all:{query}",
        max_results=400,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    year_counts = {year: 0 for year in range(start_year, current_year + 1)}
    try:
        for paper in client.results(search):
            year = paper.published.year
            if year in year_counts:
                year_counts[year] += 1
    except Exception as e:
        print("Trend Error:", e)
    return [(year, year_counts[year]) for year in sorted(year_counts)]


def search_arxiv(query, max_results=5, time_filter="All"):
    return fetch_arxiv_papers(query, max_results, time_filter)
