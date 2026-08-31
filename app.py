import json
import os
import re

import gradio as gr
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
from rapidfuzz import fuzz
from flashtext import KeywordProcessor

# -------------------------------
# Load Models (loaded once at startup)
# -------------------------------
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
model = SentenceTransformer(EMBED_MODEL_NAME)
kw_model = KeyBERT(model=model)  # reuses the same embedding model, no extra download

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def embed(text, is_query=False, **kwargs):
    if isinstance(text, str):
        text = BGE_QUERY_PREFIX + text if is_query else text
    return model.encode(text, normalize_embeddings=True, **kwargs)


# -------------------------------
# Tier 1: vetted skills taxonomy (MIND Tech Ontology, MIT licensed)
# https://github.com/MIND-TechAI/MIND-tech-ontology -- 3,333 skills with
# synonyms and implied-skill relationships, built specifically to solve
# the "React vs React.js", "Next.js implies React implies JavaScript"
# problem that a hand-written list can't cover.
# -------------------------------
TAXONOMY_PATH = os.path.join(os.path.dirname(__file__), "skills_taxonomy.json")

with open(TAXONOMY_PATH, "r") as f:
    _TAXONOMY = json.load(f)

# synonym (lowercased) -> canonical skill name
_SYNONYM_TO_CANONICAL = {}
# canonical name -> set of directly implied canonical skills
_IMPLIES = {}

for entry in _TAXONOMY:
    canonical = entry["name"]
    # Always include the canonical name itself as a matchable term -- some
    # taxonomy entries list only variant synonyms (e.g. "Java" has synonyms
    # like "JDK"/"Java8" but not the plain word "Java" itself).
    all_terms = set(entry.get("synonyms", []) or []) | {canonical}
    for syn in all_terms:
        _SYNONYM_TO_CANONICAL[syn.lower()] = canonical
    _IMPLIES[canonical] = set(entry.get("impliesKnowingSkills", []) or [])

# flashtext handles word-boundary-safe matching over thousands of terms in
# a single pass, instead of running 3,333+ individual regexes per document.
_KEYWORD_PROCESSOR = KeywordProcessor(case_sensitive=False)
for syn, canonical in _SYNONYM_TO_CANONICAL.items():
    _KEYWORD_PROCESSOR.add_keyword(syn, canonical)


def confirmed_skills(text):
    """Tier 1: match against the vetted taxonomy. Returns canonical skill
    names actually found in the text -- high precision, near-zero noise."""
    if not text:
        return set()
    return set(_KEYWORD_PROCESSOR.extract_keywords(text))


def expand_with_implied(skills):
    """If a resume states 'Next.js', credit it with 'React' and
    'JavaScript' too, since the taxonomy encodes that relationship.
    Simple closure with cycle protection."""
    expanded = set(skills)
    frontier = set(skills)
    seen = set(skills)
    for _ in range(5):  # cap depth so a cyclic graph can't loop forever
        next_frontier = set()
        for skill in frontier:
            for implied in _IMPLIES.get(skill, set()):
                if implied not in seen:
                    next_frontier.add(implied)
                    seen.add(implied)
        if not next_frontier:
            break
        expanded |= next_frontier
        frontier = next_frontier
    return expanded


# -------------------------------
# Tier 2: dynamic keyphrase extraction (gap-filler for anything the
# taxonomy doesn't know about yet -- new tools, niche jargon, typos)
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


FUZZY_MATCH_THRESHOLD = 85
SEMANTIC_MATCH_THRESHOLD = 0.55


def tier2_unverified_keywords(jd_keyphrases, resume_text, resume_keyphrases, already_confirmed_jd):
    """Only look at JD keyphrases that Tier 1 didn't already recognize as
    a taxonomy skill, so Tier 2 is purely the gap-filler, not a duplicate
    of Tier 1's work."""
    confirmed_lower = {s.lower() for s in already_confirmed_jd}
    candidates = [p for p in jd_keyphrases if not any(fuzz.ratio(p, c) > 85 for c in confirmed_lower)]
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
# Resume Quality Heuristics (general writing conventions, not a skill
# taxonomy -- these don't go stale the way a tech-skills list does)
# -------------------------------
ACTION_VERBS = [
    "led", "built", "developed", "designed", "implemented", "optimized",
    "achieved", "managed", "created", "increased", "reduced", "launched",
    "automated", "architected", "improved", "streamlined", "delivered",
    "spearheaded", "engineered", "deployed", "migrated", "mentored",
    "analyzed", "researched", "collaborated", "presented", "negotiated",
]

SECTION_KEYWORDS = {
    "experience": ["experience", "employment", "work history"],
    "education": ["education", "academic", "b.tech", "degree", "university", "college"],
    "skills": ["skills", "technical skills", "competencies"],
    "projects": ["projects", "portfolio"],
}

QUANT_PATTERN = re.compile(r"\d+(\.\d+)?\s?(%|x|\+)|\$\s?\d|\b\d{2,}\b")


def resume_quality_signals(resume_text):
    text_lower = resume_text.lower()
    lines = [l.strip() for l in resume_text.split("\n") if l.strip()]

    action_verb_lines = sum(
        1 for l in lines if any(f" {v} " in f" {l.lower()} " or l.lower().startswith(v) for v in ACTION_VERBS)
    )
    quantified_lines = sum(1 for l in lines if QUANT_PATTERN.search(l))

    sections_found = {s: any(kw in text_lower for kw in kws) for s, kws in SECTION_KEYWORDS.items()}
    sections_present = sum(sections_found.values())

    verb_score = min(action_verb_lines / max(len(lines), 1), 0.5) * 2
    quant_score = min(quantified_lines / max(len(lines), 1), 0.3) / 0.3
    section_score = sections_present / len(SECTION_KEYWORDS)

    quality_score = round((0.4 * verb_score + 0.35 * quant_score + 0.25 * section_score) * 100, 2)
    missing_sections = [s for s, present in sections_found.items() if not present]

    return {
        "quality_score": quality_score,
        "action_verb_lines": action_verb_lines,
        "quantified_lines": quantified_lines,
        "missing_sections": missing_sections,
    }


# -------------------------------
# Main Analysis
# -------------------------------
def analyze_resume(resume_text, job_description):
    if not resume_text or not job_description:
        return "Please provide both Resume and Job Description."

    # Semantic similarity (JD as query, resume as document)
    resume_emb = embed(resume_text, is_query=False, convert_to_tensor=True)
    jd_emb = embed(job_description, is_query=True, convert_to_tensor=True)
    semantic_score = round(util.cos_sim(resume_emb, jd_emb).item() * 100, 2)

    # --- Tier 1: vetted taxonomy match ---
    jd_confirmed = confirmed_skills(job_description)
    resume_confirmed_raw = confirmed_skills(resume_text)
    resume_confirmed_effective = expand_with_implied(resume_confirmed_raw)  # credit implied skills

    matched_confirmed = jd_confirmed & resume_confirmed_effective
    missing_confirmed = jd_confirmed - resume_confirmed_effective
    implied_only = {
        s for s in matched_confirmed
        if s not in resume_confirmed_raw  # matched only via an implied relationship, not stated directly
    }

    # --- Tier 2: dynamic gap-filler for anything Tier 1 didn't recognize ---
    jd_keyphrases = extract_keyphrases(job_description, top_n=15)
    resume_keyphrases = extract_keyphrases(resume_text, top_n=20)
    tier2_matched, tier2_missing = tier2_unverified_keywords(
        jd_keyphrases, resume_text, resume_keyphrases, jd_confirmed
    )

    # Overall coverage ratio blends both tiers, weighting confirmed higher
    total_jd_items = len(jd_confirmed) + len(tier2_matched) + len(tier2_missing)
    total_matched_items = len(matched_confirmed) + len(tier2_matched)
    coverage_ratio = (total_matched_items / total_jd_items) if total_jd_items else 1.0

    quality = resume_quality_signals(resume_text)

    final_score = round(
        0.45 * semantic_score
        + 0.35 * (coverage_ratio * 100)
        + 0.20 * quality["quality_score"],
        2,
    )

    # Suggestions
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

    matched_confirmed_str = ", ".join(sorted(matched_confirmed)) if matched_confirmed else "None"
    implied_note = f" (of which via implied skills: {', '.join(sorted(implied_only))})" if implied_only else ""
    missing_confirmed_str = ", ".join(sorted(missing_confirmed)) if missing_confirmed else "None"
    tier2_matched_str = ", ".join(tier2_matched) if tier2_matched else "None"
    tier2_missing_str = ", ".join(tier2_missing) if tier2_missing else "None"

    result = f"""
1. OVERALL MATCH SCORE
{final_score}%  (semantic {semantic_score}% . coverage {round(coverage_ratio * 100, 2)}% . resume quality {quality['quality_score']}%)

2. CONFIRMED SKILLS -- MATCHED (vetted taxonomy){implied_note}
{matched_confirmed_str}

3. CONFIRMED SKILLS -- MISSING (vetted taxonomy)
{missing_confirmed_str}

4. OTHER KEYWORDS DETECTED IN RESUME (unverified, AI-extracted)
{tier2_matched_str}

5. OTHER KEYWORDS MISSING (unverified, AI-extracted -- double-check these manually)
{tier2_missing_str}

6. RESUME QUALITY SIGNALS
Action-verb bullet lines: {quality['action_verb_lines']}
Quantified achievement lines: {quality['quantified_lines']}
Missing sections: {', '.join(quality['missing_sections']) if quality['missing_sections'] else 'None'}

7. SUGGESTIONS
{chr(10).join(f'- {s}' for s in suggestions)}
"""
    return result.strip()


# -------------------------------
# Gradio Interface
# -------------------------------
interface = gr.Interface(
    fn=analyze_resume,
    inputs=[
        gr.Textbox(lines=12, label="Resume Text"),
        gr.Textbox(lines=12, label="Job Description"),
    ],
    outputs=gr.Textbox(lines=26, label="Analysis Result"),
    title="AI Resume & Job Match Analyzer",
    description=(
        "Compares a resume with a job description using BGE embeddings, a vetted 3,300+ "
        "skill taxonomy (MIND Tech Ontology) for high-confidence matching, and dynamic "
        "keyphrase extraction as a gap-filler for anything the taxonomy misses."
    ),
    examples=[
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
    ],
)


if __name__ == "__main__":
    interface.launch()
