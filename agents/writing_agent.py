import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None
_LLM_MAX_WORKERS = 6


def _call_llm(prompt, temperature=0.3, max_tokens=700):
    if not client:
        return ""
    try:
        return client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        ).choices[0].message.content.strip()
    except Exception:
        return ""


def _build_context_block(analyses, insights, gaps, recommendations):
    block = ""
    uploaded = [analysis for analysis in analyses if analysis.get("title", "").startswith("[Uploaded]")]
    others = [analysis for analysis in analyses if not analysis.get("title", "").startswith("[Uploaded]")]
    selected = (uploaded + others)[:6]
    for analysis in selected:
        block += (
            f"Paper: {analysis.get('title', '')}\n"
            f"Problem: {analysis.get('problem', '')}\n"
            f"Method: {analysis.get('method', '')}\n"
            f"Dataset: {analysis.get('dataset', '')}\n"
            f"Performance: {analysis.get('performance', '')}\n"
            f"Limitations: {analysis.get('limitations', '')}\n\n"
        )
    block += f"Cross-paper insights:\n{insights}\n\n"
    block += f"Research gaps:\n{gaps}\n\n"
    block += f"Recommendations:\n{recommendations}\n"
    return block[:3500]


def _run_llm_task(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def live_analyze(user_text, section, analyses, insights, gaps, recommendations):
    if not user_text or len(user_text.strip()) < 10:
        return _empty_result()

    if not client:
        result = _empty_result()
        result["section_guide"] = _section_guide(section)
        result["redundancy"] = _redundancy_check(user_text.strip())
        return result

    context = _build_context_block(analyses, insights, gaps, recommendations)
    text = user_text.strip()

    llm_jobs = [
        ("suggestions", _live_suggestions, (text, section, context)),
        ("context_hints", _context_aware_hints, (text, context)),
        ("autocomplete", _live_autocomplete, (text, context)),
        ("gap_alignment", _gap_alignment, (text, gaps)),
        ("style_improved", _style_improvement, (text, section)),
        ("flow_issue", _flow_check, (text, section)),
        ("experiment_suggestion", _experiment_suggestion, (text, context)),
        ("metric_suggestion", _metric_suggestion, (text, context)),
        ("novel_idea", _novel_idea, (text, context)),
    ]

    out = {k: None for k, _, _ in llm_jobs}
    workers = min(_LLM_MAX_WORKERS, len(llm_jobs))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(_run_llm_task, fn, *args): key
            for key, fn, args in llm_jobs
        }
        for fut in as_completed(future_map):
            key = future_map[fut]
            try:
                out[key] = fut.result()
            except Exception:
                out[key] = None

    return {
        "suggestions": out["suggestions"] or [],
        "context_hints": out["context_hints"] or [],
        "autocomplete": out["autocomplete"] or "",
        "gap_alignment": out["gap_alignment"] or "",
        "style_improved": out["style_improved"] or "",
        "redundancy": _redundancy_check(text),
        "flow_issue": out["flow_issue"],
        "experiment_suggestion": out["experiment_suggestion"] or "",
        "metric_suggestion": out["metric_suggestion"] or "",
        "novel_idea": out["novel_idea"] or "",
        "section_guide": _section_guide(section),
    }


def improve_paragraph(user_text, section, analyses, insights, gaps, recommendations):
    if not user_text:
        return ""
    if not client:
        return user_text

    context = _build_context_block(analyses, insights, gaps, recommendations)
    prompt = f"""You are an expert academic writing assistant.

Section: {section}
Research context:
{context}

Original paragraph:
\"\"\"{user_text}\"\"\"

Rewrite this paragraph to be more precise, academic, and grounded in the research context.
- Fix vague claims with specific evidence or metrics where the context supports them.
- Improve logical flow.
- Do not add unsupported facts or generic filler.
Return only the improved paragraph."""
    return _call_llm(prompt, temperature=0.4, max_tokens=600) or user_text


def _live_suggestions(text, section, context):
    prompt = f"""You are a research writing coach. Section: {section}.

Research context:
{context}

User wrote:
\"{text}\"

Give 3 short, specific suggestions grounded in the research context.
Each suggestion must start with "-".
Do not give generic advice."""
    raw = _call_llm(prompt, temperature=0.3, max_tokens=280)
    bullets = [line.strip().lstrip("-").strip() for line in raw.splitlines() if line.strip().startswith("-")]
    return bullets[:3] if bullets else ([raw] if raw else [])


def _context_aware_hints(text, context):
    prompt = f"""Retrieved paper context:
{context}

User wrote:
\"{text}\"

List up to 2 specific points from the retrieved papers that support, contradict, or extend the user's claim.
Each line must start with "-". Be specific."""
    raw = _call_llm(prompt, temperature=0.2, max_tokens=220)
    hints = [line.strip().lstrip("-").strip() for line in raw.splitlines() if line.strip().startswith("-")]
    return hints[:2]


def _live_autocomplete(text, context):
    triggers = {"because", "since", "however", "although", "therefore", "lack", "fails", "whereas"}
    words = text.lower().split()
    if not any(word.strip(",.") in triggers for word in words[-5:]) and not text.rstrip().endswith(","):
        return ""

    prompt = f"""Research context:
{context}

The user stopped here:
\"{text}\"

Continue only the current sentence in 1-2 lines. Do not repeat the user's text."""
    return _call_llm(prompt, temperature=0.4, max_tokens=100)


def _gap_alignment(text, gaps):
    if not gaps or gaps.strip().lower() == "no data available.":
        return ""

    prompt = f"""Research gaps:
{gaps}

User wrote:
\"{text}\"

Does the user's text align with any identified research gap?
One line only."""
    return _call_llm(prompt, temperature=0.2, max_tokens=100)


def _style_improvement(text, section):
    prompt = f"""Rewrite this text to sound more academic and precise.
Section: {section}
Text:
\"{text}\"

Return only the improved version. Keep it to 1-2 sentences."""
    return _call_llm(prompt, temperature=0.3, max_tokens=140)


def _redundancy_check(text):
    words = re.findall(r"\b\w{5,}\b", text.lower())
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1

    repeated = [word for word, count in freq.items() if count >= 3]
    if repeated:
        return f"Repeated terms detected: {', '.join(repeated[:3])}. Consider varying your wording."

    phrases = re.findall(r"\b\w+ \w+\b", text.lower())
    phrase_freq = {}
    for phrase in phrases:
        phrase_freq[phrase] = phrase_freq.get(phrase, 0) + 1

    repeated_phrases = [phrase for phrase, count in phrase_freq.items() if count >= 2 and len(phrase) > 8]
    if repeated_phrases:
        return f'Repeated phrase: "{repeated_phrases[0]}". Try paraphrasing.'

    return None


def _flow_check(text, section):
    prompt = f"""Section: {section}
Text:
\"{text}\"

Check only for logical flow issues. If there is an issue, start with "Flow issue -".
If no issue, return "OK"."""
    result = _call_llm(prompt, temperature=0.2, max_tokens=70)
    return None if result.strip().upper().startswith("OK") else result.strip()


def _experiment_suggestion(text, context):
    prompt = f"""Research context:
{context}

User is writing about:
\"{text}\"

Suggest one specific experiment or comparison this work could include, referencing methods or datasets from the papers.
One line only."""
    return _call_llm(prompt, temperature=0.4, max_tokens=100)


def _metric_suggestion(text, context):
    prompt = f"""Research context:
{context}

User is writing:
\"{text}\"

List 3-5 relevant evaluation metrics drawn from the retrieved papers.
One line only."""
    return _call_llm(prompt, temperature=0.2, max_tokens=90)


def _novel_idea(text, context):
    prompt = f"""Research context:
{context}

User is writing about:
\"{text}\"

Suggest one novel research idea that combines or extends concepts in the retrieved papers.
One line only."""
    return _call_llm(prompt, temperature=0.7, max_tokens=120)


def _section_guide(section):
    guides = {
        "Introduction": "Start broad, narrow to the gap, and end with clear contributions.",
        "Related Work": "Group papers by theme and compare methods, datasets, and limitations.",
        "Methodology": "Define the design choices, variables, datasets, and evaluation metrics.",
        "Conclusion": "Summarize contributions, acknowledge limitations, and state concrete future work.",
    }
    return guides.get(section, "")


def _empty_result():
    return {
        "suggestions": [],
        "context_hints": [],
        "autocomplete": "",
        "gap_alignment": "",
        "style_improved": "",
        "redundancy": None,
        "flow_issue": None,
        "experiment_suggestion": "",
        "metric_suggestion": "",
        "novel_idea": "",
        "section_guide": "",
    }
