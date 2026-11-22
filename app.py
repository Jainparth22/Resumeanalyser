"""AI Resume & Job Match Analyzer -- Gradio UI + public API.

Pure logic lives in the `resume_analyser` package (tested without ML models);
this file wires it to the embedding model, KeyBERT and the Gradio interface.
"""

import os
import re

import gradio as gr
import numpy as np
from keybert import KeyBERT
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

from resume_analyser import __version__
from resume_analyser.quality import resume_quality_signals
from resume_analyser.ratelimit import RateLimitExceeded, SlidingWindowRateLimiter
from resume_analyser.taxonomy import load_default_taxonomy

# -------------------------------
# Configuration (override with Space "Variables" / environment variables)
# -------------------------------
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "20000"))
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_WINDOW_SECONDS = float(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
CHUNK_CHARS = 1200  # ~300 tokens, safely under BGE's 512-token window

FUZZY_MATCH_THRESHOLD = 85
SEMANTIC_MATCH_THRESHOLD = 0.55
# Raw BGE cosine similarity of related texts clusters in ~0.45-0.85, so using
# it directly as a percentage under-reports good matches. These bounds are a
# heuristic linear rescale -- tune them against labelled resume/JD pairs.
COSINE_FLOOR = 0.45
COSINE_CEIL = 0.85

# -------------------------------
# Models (loaded once at startup)
# -------------------------------
model = SentenceTransformer(EMBED_MODEL_NAME)
kw_model = KeyBERT(model=model)  # reuses the same embedding model
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

taxonomy = load_default_taxonomy()
rate_limiter = SlidingWindowRateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)


def embed(text, is_query=False, **kwargs):
    if isinstance(text, str) and is_query:
        text = BGE_QUERY_PREFIX + text
    return model.encode(text, normalize_embeddings=True, **kwargs)


def split_chunks(text, max_chars=CHUNK_CHARS):
    """Greedily pack lines into chunks so nothing is lost to 512-token truncation."""
    chunks, current = [], ""
    for line in (l.strip() for l in text.split("\n")):
        if not line:
            continue
        while len(line) > max_chars:  # very long single line
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:max_chars])
            line = line[max_chars:]
        if current and len(current) + len(line) + 1 > max_chars:
            chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)
    return chunks or [text]


def embed_long(text, is_query=False):
    """Mean-pool chunk embeddings into one unit vector for the whole document."""
    chunks = split_chunks(text)
    if is_query:
        chunks = [BGE_QUERY_PREFIX + c for c in chunks]
    vecs = model.encode(chunks, normalize_embeddings=True)
    mean = np.asarray(vecs).mean(axis=0)
    return mean / (np.linalg.norm(mean) or 1.0)


def calibrate_semantic(cosine):
    """Linear rescale of raw cosine similarity to a 0-100 score."""
    scaled = (cosine - COSINE_FLOOR) / (COSINE_CEIL - COSINE_FLOOR)
    return round(min(max(scaled, 0.0), 1.0) * 100, 2)


# -------------------------------
# Tier 2: dynamic keyphrase extraction (gap-filler for what the taxonomy
# doesn't know -- new tools, niche jargon, typos)
# -------------------------------
def extract_keyphrases(text, top_n=15):
    if not text or not text.strip():
        return []
    try:
        raw = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=top_n,
            use_mmr=True,
            diversity=0.6,
        )
    except Exception:
        return []
    return [phrase.strip().lower() for phrase, _ in raw if len(phrase.strip()) >= 2]


def tier2_unverified_keywords(jd_keyphrases, resume_text, resume_keyphrases, already_confirmed_jd):
    """Only JD keyphrases Tier 1 didn't already recognise, so Tier 2 is purely
    the gap-filler and never double-counts Tier 1."""
    confirmed_lower = {s.lower() for s in already_confirmed_jd}
    candidates = [
        p for p in jd_keyphrases
        if not any(fuzz.ratio(p, c) > FUZZY_MATCH_THRESHOLD for c in confirmed_lower)
    ]
    if not candidates:
        return [], []

    resume_norm = resume_text.lower().replace("-", " ").replace(".", "")
    jd_embeddings = embed(candidates, convert_to_tensor=True)
    resume_phrase_embeddings = embed(resume_keyphrases, convert_to_tensor=True) if resume_keyphrases else None

    matched, missing = [], []
    for i, phrase in enumerate(candidates):
        phrase_norm = phrase.replace("-", " ").replace(".", "")
        if fuzz.partial_ratio(phrase_norm, resume_norm) >= FUZZY_MATCH_THRESHOLD:
            matched.append(phrase)
            continue
        if resume_phrase_embeddings is not None:
            sims = util.cos_sim(jd_embeddings[i], resume_phrase_embeddings)[0]
            if sims.max().item() >= SEMANTIC_MATCH_THRESHOLD:
                matched.append(phrase)
                continue
        missing.append(phrase)
    return matched, missing


# -------------------------------
# Core analysis (returns a JSON-serialisable dict -- this is the API contract)
# -------------------------------
def analyze(resume_text, job_description):
    cosine = float(np.dot(embed_long(resume_text), embed_long(job_description, is_query=True)))
    semantic_score = calibrate_semantic(cosine)

    # Tier 1: vetted taxonomy
    jd_confirmed = taxonomy.confirmed_skills(job_description)
    resume_raw = taxonomy.confirmed_skills(resume_text)
    resume_effective = taxonomy.expand_with_implied(resume_raw)
    matched_confirmed = jd_confirmed & resume_effective
    missing_confirmed = jd_confirmed - resume_effective
    implied_only = {s for s in matched_confirmed if s not in resume_raw}

    # Tier 2: gap-filler
    jd_keyphrases = extract_keyphrases(job_description, top_n=15)
    resume_keyphrases = extract_keyphrases(resume_text, top_n=20)
    tier2_matched, tier2_missing = tier2_unverified_keywords(
        jd_keyphrases, resume_text, resume_keyphrases, jd_confirmed
    )

    total_items = len(jd_confirmed) + len(tier2_matched) + len(tier2_missing)
    matched_items = len(matched_confirmed) + len(tier2_matched)
    coverage_ratio = (matched_items / total_items) if total_items else 1.0

    quality = resume_quality_signals(resume_text)
    final_score = round(
        0.45 * semantic_score + 0.35 * (coverage_ratio * 100) + 0.20 * quality["quality_score"], 2
    )

    suggestions = []
    if missing_confirmed:
        suggestions.append(
            f"Confirmed JD requirements missing from your resume: {', '.join(sorted(missing_confirmed)[:8])}."
        )
    if tier2_missing:
        suggestions.append(
            f"Also worth checking (auto-detected, not taxonomy-verified): {', '.join(tier2_missing[:5])}."
        )
    if quality["missing_sections"]:
        suggestions.append(
            f"Resume appears to be missing a clear '{', '.join(quality['missing_sections'])}' section -- "
            "ATS parsers rely on section headers to categorize content."
        )
    if quality["quantified_lines"] == 0:
        suggestions.append(
            "No quantified achievements detected (numbers, %, $, scale). Try rephrasing bullets like "
            "'Improved pipeline performance' as 'Improved pipeline throughput by 30%'."
        )
    if not suggestions:
        suggestions.append("Strong match -- resume aligns well with this job description.")

    return {
        "version": __version__,
        "scores": {
            "overall": final_score,
            "semantic": semantic_score,
            "semantic_raw_cosine": round(cosine, 4),
            "coverage": round(coverage_ratio * 100, 2),
            "quality": quality["quality_score"],
        },
        "confirmed_skills": {
            "matched": sorted(matched_confirmed),
            "missing": sorted(missing_confirmed),
            "matched_via_implication": sorted(implied_only),
        },
        "other_keywords": {"matched": tier2_matched, "missing": tier2_missing},
        "quality_signals": {
            "action_verb_lines": quality["action_verb_lines"],
            "quantified_lines": quality["quantified_lines"],
            "missing_sections": quality["missing_sections"],
        },
        "suggestions": suggestions,
    }


def format_report(r):
    s, c, k, q = r["scores"], r["confirmed_skills"], r["other_keywords"], r["quality_signals"]
    join = lambda items: ", ".join(items) if items else "None"
    implied_note = (
        f" (of which via implied skills: {', '.join(c['matched_via_implication'])})"
        if c["matched_via_implication"]
        else ""
    )
    return f"""
1. OVERALL MATCH SCORE
{s['overall']}%  (semantic {s['semantic']}% . coverage {s['coverage']}% . resume quality {s['quality']}%)

2. CONFIRMED SKILLS -- MATCHED (vetted taxonomy){implied_note}
{join(c['matched'])}

3. CONFIRMED SKILLS -- MISSING (vetted taxonomy)
{join(c['missing'])}

4. OTHER KEYWORDS DETECTED IN RESUME (unverified, AI-extracted)
{join(k['matched'])}

5. OTHER KEYWORDS MISSING (unverified, AI-extracted -- double-check these manually)
{join(k['missing'])}

6. RESUME QUALITY SIGNALS
Action-verb bullet lines: {q['action_verb_lines']}
Quantified achievement lines: {q['quantified_lines']}
Missing sections: {join(q['missing_sections'])}

7. SUGGESTIONS
{chr(10).join(f'- {x}' for x in r['suggestions'])}
""".strip()


# -------------------------------
# Request handling: validation + rate limiting
# -------------------------------
def client_key(request):
    """Best-effort client identity. On HF Spaces the proxy sets X-Forwarded-For;
    outside a trusted proxy that header is client-controlled (see docs)."""
    if request is None:
        return "anonymous"
    forwarded = request.headers.get("x-forwarded-for") if request.headers else None
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "anonymous"


def handle_request(resume_text, job_description, request: gr.Request = None):
    if not resume_text or not resume_text.strip() or not job_description or not job_description.strip():
        raise gr.Error("Please provide both Resume and Job Description.")
    if len(resume_text) > MAX_INPUT_CHARS or len(job_description) > MAX_INPUT_CHARS:
        raise gr.Error(f"Input too long (max {MAX_INPUT_CHARS} characters each).")
    try:
        rate_limiter.check(client_key(request))
    except RateLimitExceeded as e:
        raise gr.Error(
            f"Rate limit exceeded ({RATE_LIMIT_REQUESTS} requests / {int(RATE_LIMIT_WINDOW_SECONDS)}s). "
            f"Try again in {e.retry_after:.0f}s."
        )
    result = analyze(resume_text, job_description)
    return format_report(result), result


# -------------------------------
# Gradio UI (the same click handler is exposed as the public API: /analyze)
# -------------------------------
EXAMPLES = [
    [
        """Name: Ananya Verma

Summary:
Data Analyst with experience in data processing and visualization, looking to transition into machine learning roles.

Skills:
Python, SQL, Pandas, NumPy, Data Science, Scikit-learn, Git

Experience:
- Performed data cleaning and analysis using Python and SQL, improving report turnaround by 20%.
- Built regression and classification models using scikit-learn.
- Created dashboards and reports for business stakeholders.

Education:
M.Sc in Data Science
""",
        """Job Title: NLP / Machine Learning Engineer

Requirements:
- Strong experience in Python and Machine Learning
- Hands-on experience with NLP and Transformers
- Familiarity with Hugging Face libraries
- Experience with PyTorch
- Knowledge of SQL and data processing tools
- Experience deploying ML models is a plus
""",
    ],
    [
        "Experienced Python developer with strong background in machine learning, NLP, and data science.",
        "Looking for a data scientist skilled in Python, machine learning, NLP, SQL, and Hugging Face.",
    ],
]

with gr.Blocks(title="AI Resume & Job Match Analyzer") as demo:
    gr.Markdown(
        "# AI Resume & Job Match Analyzer\n"
        "Compares a resume with a job description using BGE embeddings, a vetted 3,300+ skill "
        "taxonomy (MIND Tech Ontology) for high-confidence matching, and KeyBERT keyphrase "
        "extraction as a gap-filler. Open source (MIT) -- free API usage: see the "
        "**Use via API** link in the footer."
    )
    with gr.Row():
        resume_in = gr.Textbox(lines=12, label="Resume Text")
        jd_in = gr.Textbox(lines=12, label="Job Description")
    run_btn = gr.Button("Analyze", variant="primary")
    report_out = gr.Textbox(lines=26, label="Analysis Result")
    with gr.Accordion("Structured JSON (same payload the API returns)", open=False):
        json_out = gr.JSON(label="JSON")
    gr.Examples(examples=EXAMPLES, inputs=[resume_in, jd_in])
    run_btn.click(
        handle_request,
        inputs=[resume_in, jd_in],
        outputs=[report_out, json_out],
        api_name="analyze",
    )

demo.queue(default_concurrency_limit=2, max_size=20)

if __name__ == "__main__":
    demo.launch()
