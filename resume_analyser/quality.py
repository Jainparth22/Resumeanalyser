"""Resume writing-quality heuristics (general conventions, not a skills list)."""

import re

# Stems, so "build", "builds", "built", "building" are all recognised.
ACTION_VERB_PATTERN = re.compile(
    r"\b(?:led|lead(?:s|ing)?|built|build(?:s|ing)?|develop(?:ed|s|ing)?|"
    r"design(?:ed|s|ing)?|implement(?:ed|s|ing)?|optimi[sz](?:ed|es|ing)?|"
    r"achiev(?:ed|es|ing)?|manag(?:ed|es|ing)?|creat(?:ed|es|ing)?|"
    r"increas(?:ed|es|ing)?|reduc(?:ed|es|ing)?|launch(?:ed|es|ing)?|"
    r"automat(?:ed|es|ing)?|architect(?:ed|s|ing)?|improv(?:ed|es|ing)?|"
    r"streamlin(?:ed|es|ing)?|deliver(?:ed|s|ing)?|spearhead(?:ed|s|ing)?|"
    r"engineer(?:ed|s|ing)?|deploy(?:ed|s|ing)?|migrat(?:ed|es|ing)?|"
    r"mentor(?:ed|s|ing)?|analy[sz](?:ed|es|ing)?|research(?:ed|es|ing)?|"
    r"collaborat(?:ed|es|ing)?|present(?:ed|s|ing)?|negotiat(?:ed|es|ing)?|"
    r"contribut(?:ed|es|ing)?|publish(?:ed|es|ing)?|train(?:ed|s|ing)?)\b",
    re.IGNORECASE,
)

SECTION_KEYWORDS = {
    "experience": ["experience", "employment", "work history"],
    "education": ["education", "academic", "b.tech", "degree", "university", "college"],
    "skills": ["skills", "technical skills", "competencies"],
    "projects": ["projects", "portfolio"],
}

# Counts achievements with a *metric* (45%, 3x, +20, $2M, 10k users, 200 ms),
# but not bare numbers such as years (2023), phone numbers or zip codes.
QUANT_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s?(?:%|x\b|\+)"
    r"|[$₹€£]\s?\d"
    r"|\b\d[\d,]*(?:\.\d+)?\s?(?:k|m|b|ms|s|gb|mb|users|customers|requests|records|"
    r"models|projects|members|engineers|hours|days|people|clients|endpoints)\b",
    re.IGNORECASE,
)


def resume_quality_signals(resume_text):
    text_lower = resume_text.lower()
    lines = [l.strip() for l in resume_text.split("\n") if l.strip()]
    n = max(len(lines), 1)

    action_verb_lines = sum(1 for l in lines if ACTION_VERB_PATTERN.search(l))
    quantified_lines = sum(1 for l in lines if QUANT_PATTERN.search(l))

    sections_found = {s: any(kw in text_lower for kw in kws) for s, kws in SECTION_KEYWORDS.items()}
    sections_present = sum(sections_found.values())

    verb_score = min(action_verb_lines / n, 0.5) * 2
    quant_score = min(quantified_lines / n, 0.3) / 0.3
    section_score = sections_present / len(SECTION_KEYWORDS)

    quality_score = round((0.4 * verb_score + 0.35 * quant_score + 0.25 * section_score) * 100, 2)
    missing_sections = [s for s, present in sections_found.items() if not present]

    return {
        "quality_score": quality_score,
        "action_verb_lines": action_verb_lines,
        "quantified_lines": quantified_lines,
        "missing_sections": missing_sections,
    }
