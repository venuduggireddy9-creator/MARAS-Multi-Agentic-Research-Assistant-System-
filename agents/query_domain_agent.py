import re
from collections import defaultdict


def paper_matches_domain(domain: str, text: str) -> bool:
    """Word-boundary style checks for domain boost / filter (avoids 'ml' in 'html')."""
    text_l = text.lower()
    patterns = {
        "security": r"\b(?:intrusion|malware|cyber|network security|firewall|attack detection)\b",
        "xai": r"\b(?:explainable|interpretability|xai|lime|shap)\b",
        "ml": r"\b(?:machine learning|deep learning|neural network|transformer|cnn|rnn|gnn)\b",
        "healthcare": r"\b(?:medical|clinical|healthcare|patient|diagnosis|hospital)\b",
        "vision": r"\b(?:image segmentation|computer vision|object detection|lidar|camera)\b",
        "blockchain": r"\b(?:blockchain|smart contract|ledger)\b",
        "distributed": r"\b(?:distributed system|consensus|replication|paxos|raft|byzantine)\b",
        "nlp": r"\b(?:nlp|natural language|language model|bert|tokenization)\b",
    }
    pat = patterns.get(domain)
    if pat and re.search(pat, text_l):
        return True
    if domain == "ml" and re.search(r"\bml\b", text_l):
        return True
    return False


def refine_query(query: str):
    q = query.lower().strip()
    words = list(dict.fromkeys(q.split()))
    return " ".join(words)


def generate_subqueries(query: str, max_queries: int = 3):
    """
    Focused sub-queries for arXiv retrieval. Capped for latency.
    Single source of truth — retrieval_filter_agent imports this.
    """
    q = query.lower()
    subs = [query.strip()]
    has_health = any(k in q for k in ("clinical", "healthcare", "medical", "patient", "diagnosis", "hospital"))

    if "fairness" in q or "bias" in q:
        if has_health:
            subs.append("fairness machine learning clinical prediction bias")
            subs.append("algorithmic bias healthcare machine learning")
        else:
            subs.append("algorithmic fairness machine learning classification")
            subs.append("bias mitigation deep learning fairness benchmarks")

    if has_health:
        subs.append("clinical decision support machine learning healthcare")
        subs.append("medical diagnosis deep learning prediction models")

    if "explainable" in q or "interpretability" in q or "xai" in q:
        subs.append("explainable ai interpretability machine learning")
        subs.append("model explanation neural network interpretability")

    if "intrusion" in q or _ids_token(q):
        subs.append("intrusion detection machine learning network security")
        subs.append("network anomaly detection deep learning cybersecurity")

    if "federated" in q:
        subs.append("federated learning privacy distributed optimization")
        if has_health:
            subs.append("federated learning healthcare medical imaging")

    if "blockchain" in q:
        subs.append("blockchain distributed ledger consensus smart contracts")

    if "autonomous" in q or "vehicle" in q or "self-driving" in q:
        subs.append("autonomous driving perception sensor fusion deep learning")
        subs.append("object detection lidar camera autonomous vehicles")

    if "compiler" in q or "program analysis" in q:
        subs.append("compiler optimization program analysis static analysis")

    if "reinforcement" in q or "rl " in q or q.strip() == "rl":
        subs.append("reinforcement learning deep learning policy optimization")

    if "graph neural" in q or " gnn" in q or q.endswith("gnn"):
        subs.append("graph neural networks message passing representation learning")

    unique = list(dict.fromkeys(subs))
    return unique[: max(1, max_queries)]


def detect_domains(text: str):
    text = text.lower()
    scores = defaultdict(int)
    domains = set()

    if any(k in text for k in [
        "intrusion", "intrusion detection", "cyber attack", "malware",
        "network security", "anomaly detection",
    ]) or _ids_token(text):
        domains.add("security")

    if any(k in text for k in [
        "explainable", "interpretability", "explanation", "transparent", "xai",
    ]):
        domains.add("xai")

    DOMAIN_KEYWORDS = {
        "distributed": [
            "distributed", "consensus", "replication", "byzantine", "paxos", "raft",
        ],
        "ml": [
            "machine learning", "deep learning", "neural", "transformer", "classification",
        ],
        "healthcare": [
            "healthcare", "medical", "clinical", "diagnosis", "patient",
        ],
        "blockchain": [
            "blockchain", "ledger", "smart contract", "decentralized",
        ],
        "vision": [
            "image", "video", "vision", "mri", "ct", "radiology", "segmentation",
        ],
        "nlp": [
            "language model", "nlp", "bert", "tokenization", "summarization",
        ],
    }

    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[domain] += 2 if len(kw.split()) > 1 else 1

    for d, s in scores.items():
        if s >= 2:
            domains.add(d)

    if not domains:
        fallback = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
        domains = {d for d, s in fallback if s > 0}

    return list(domains)


def _ids_token(q: str) -> bool:
    return "ids" in set(q.replace(",", " ").split())
