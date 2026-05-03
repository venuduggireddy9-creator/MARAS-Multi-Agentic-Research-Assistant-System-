import os
import re
from collections import Counter

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None


def _comparison_table(analyses):
    return [
        {
            "title": a.get("title", ""),
            "method": a.get("method", "Not specified"),
            "dataset": a.get("dataset", "Not specified"),
            "performance": a.get("performance", "Not specified"),
            "application": a.get("application", "Not specified"),
            "limitations": a.get("limitations", "Not specified"),
        }
        for a in analyses
    ]


def _clean_value(value):
    value = (value or "").strip()
    return "" if value.lower() in {"", "not specified", "none", "n/a"} else value


def _short_title(title):
    title = (title or "Untitled paper").strip()
    return title[:85] + "..." if len(title) > 85 else title


def _tokens(values):
    text = " ".join(_clean_value(v).lower() for v in values)
    words = re.findall(r"[a-z][a-z0-9\-]{3,}", text)
    stop = {
        "paper", "study", "method", "model", "approach", "using", "based", "proposed",
        "results", "dataset", "performance", "system", "this", "that", "with", "from",
    }
    return [w for w in words if w not in stop]


def _data_driven_reasoning(analyses):
    count = len(analyses)
    methods = [_clean_value(a.get("method")) for a in analyses]
    datasets = [_clean_value(a.get("dataset")) for a in analyses]
    performances = [_clean_value(a.get("performance")) for a in analyses]
    limitations = [_clean_value(a.get("limitations")) for a in analyses]
    applications = [_clean_value(a.get("application")) for a in analyses]

    method_terms = [term for term, _ in Counter(_tokens(methods)).most_common(5)]
    app_terms = [term for term, _ in Counter(_tokens(applications)).most_common(4)]
    named_datasets = [d for d in datasets if d]
    reported_perf = [p for p in performances if p]
    reported_limits = [l for l in limitations if l]

    insights = []

    if count >= 2:
        method_focus = ", ".join(method_terms[:3]) if method_terms else "different modeling strategies"
        insights.append(
            f"Across {count} papers, the method landscape is not uniform: recurring method signals such as {method_focus} suggest that the literature is exploring multiple solution families rather than converging on one standard design."
        )

    if named_datasets:
        unique_datasets = sorted(set(named_datasets))
        if len(unique_datasets) == 1:
            insights.append(
                f"Dataset usage is concentrated around {unique_datasets[0]}, which makes results easier to compare but leaves open whether the methods generalize beyond that benchmark."
            )
        else:
            insights.append(
                f"The retrieved papers report different datasets or evaluation settings ({'; '.join(unique_datasets[:3])}), so performance claims should be compared cautiously because benchmark inconsistency can hide method-level trade-offs."
            )
    else:
        insights.append(
            "Dataset reporting is weak across the selected papers; this limits reproducibility and makes it difficult to build a reliable comparison table for a literature review."
        )

    if reported_perf:
        insights.append(
            f"Performance evidence is uneven: {len(reported_perf)} of {count} papers mention metrics or improvements, while the rest provide limited quantitative grounding, creating an evaluation gap for research writing."
        )
    else:
        insights.append(
            "None of the extracted analyses provide concrete performance values, so any literature review should avoid ranking the methods by effectiveness until metrics are recovered from the full papers."
        )

    if reported_limits:
        insights.append(
            f"Limitations are explicitly visible in {len(reported_limits)} papers, which is useful for gap analysis because these constraints can be converted into future-work directions rather than treated as isolated weaknesses."
        )
    else:
        insights.append(
            "The papers do not clearly state limitations in the extracted text, indicating a need to inspect discussion and conclusion sections before finalizing research gaps."
        )

    if app_terms:
        insights.append(
            f"Research focus clusters around terms such as {', '.join(app_terms[:3])}; papers outside this cluster may be useful as contrast cases because they test whether the same methods transfer across tasks."
        )

    gaps = []
    if not named_datasets or len(set(named_datasets)) > 1:
        gaps.append("Standardized dataset reporting is missing or inconsistent, reducing comparability across papers.")
    if len(reported_perf) < count:
        gaps.append("Several papers lack concrete metrics in the extracted analysis, weakening evidence-based comparison.")
    if len(reported_limits) < count:
        gaps.append("Limitations and future-work constraints are not consistently stated, making gap identification less direct.")
    if not gaps:
        gaps.append("The main remaining gap is cross-benchmark validation beyond the reported experimental settings.")

    recommendations = [
        "Use a comparison matrix that normalizes method, dataset, metric, and limitation fields before making claims.",
        "Prioritize papers with explicit datasets and metrics when discussing empirical strength.",
        "Frame future work around unresolved limitations and missing evaluation settings found across multiple papers.",
    ]

    return "\n".join(f"- {item}" for item in insights[:6]), "\n".join(f"- {item}" for item in gaps), "\n".join(f"- {item}" for item in recommendations)


def _parse_sections(text):
    sections = {"insights": [], "gaps": [], "recommendations": []}
    current = None

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        header = line.strip(":").lower()
        if header.startswith("insights"):
            current = "insights"
            continue
        if header.startswith("gaps"):
            current = "gaps"
            continue
        if header.startswith("recommendations"):
            current = "recommendations"
            continue

        if current:
            sections[current].append(line)

    return (
        "\n".join(sections["insights"]).strip(),
        "\n".join(sections["gaps"]).strip(),
        "\n".join(sections["recommendations"]).strip(),
    )


def _valid_insights(insights, analyses):
    if not insights:
        return False
    lower = insights.lower()
    comparison_signals = ["while", "whereas", "however", "across", "compared", "in contrast", "trade-off"]
    data_signals = [_short_title(a.get("title", "")).split()[0].lower() for a in analyses if a.get("title")]
    return any(s in lower for s in comparison_signals) and (len(insights.split()) >= 60 or any(s in lower for s in data_signals))


def generate_insights(analyses):
    comparison = _comparison_table(analyses)

    if not analyses:
        insights, gaps, recs = _data_driven_reasoning([])
        return comparison, insights, gaps, recs

    if client is None:
        insights, gaps, recs = _data_driven_reasoning(analyses)
        return comparison, insights, gaps, recs

    paper_block = ""
    for idx, a in enumerate(analyses, start=1):
        paper_block += f"""
Paper {idx}: {_short_title(a.get('title'))}
Problem: {a.get('problem', 'Not specified')}
Method: {a.get('method', 'Not specified')}
Dataset: {a.get('dataset', 'Not specified')}
Performance: {a.get('performance', 'Not specified')}
Application: {a.get('application', 'Not specified')}
Limitations: {a.get('limitations', 'Not specified')}
"""

    prompt = f"""
You are generating literature-review intelligence from extracted paper analyses.

Use ONLY the extracted data below. Compare ALL papers together; do not summarize them one by one.

Output exactly:
INSIGHTS:
- 4 to 6 strong cross-paper insights. Each insight must be specific and useful for research writing.
- Cover method diversity, dataset consistency, evaluation/performance trends, research focus differences, and trade-offs.
- Avoid generic lines such as "various approaches are used".

GAPS:
- 3 to 5 research gaps derived from multiple papers.

RECOMMENDATIONS:
- 3 to 5 actionable recommendations tied to the gaps.

Extracted analyses:
{paper_block}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
            max_tokens=900,
        )
        insights, gaps, recs = _parse_sections(response.choices[0].message.content)
        if _valid_insights(insights, analyses):
            return comparison, insights, gaps, recs
    except Exception as exc:
        print("Insight generation error:", exc)

    insights, gaps, recs = _data_driven_reasoning(analyses)
    return comparison, insights, gaps, recs


def build_table(papers):
    table = []
    for paper in papers:
        fields = paper.get("fields", {})
        table.append(
            {
                "Rank": paper.get("rank", ""),
                "Title": paper.get("title", "")[:50],
                "Method": fields.get("method", ""),
                "Dataset": fields.get("dataset", ""),
                "Performance": fields.get("performance", ""),
            }
        )
    return table
