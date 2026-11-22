"""Split a job description into requirements and a resume into evidence bullets.

Model-free. The matcher compares these units one by one, which is what lets
the analyzer say *which* requirement is unmet and *which* resume line is the
closest evidence, instead of only returning one document-level score.
"""

import re

_BULLET_PREFIX = re.compile(r"^\s*(?:[-*•●▪–—]|\d+[.)])\s*")
_MIN_WORDS = 3
MAX_UNITS = 40


def _clean(line):
    return _BULLET_PREFIX.sub("", line).strip()


def _is_heading(line):
    # "Requirements:" / "Experience:" -- a label with nothing after the colon.
    return line.endswith(":") or (len(line.split()) <= 2 and line.isupper())


def _units(text, split_sentences=False):
    units = []
    for raw in text.split("\n"):
        line = _clean(raw)
        if not line or _is_heading(line):
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", line) if split_sentences else [line]
        for part in parts:
            part = part.strip()
            if len(part.split()) >= _MIN_WORDS:
                units.append(part)
    return units[:MAX_UNITS]


def split_requirements(job_description):
    """One requirement per bullet/line; paragraphs are split into sentences.
    A 'Job Title: ...' line is dropped: it is a label, not a requirement."""
    lines = [l for l in job_description.split("\n") if not re.match(r"^\s*job title\s*:", l, re.I)]
    return _units("\n".join(lines), split_sentences=True)


def split_bullets(resume_text):
    """One evidence unit per resume line (bullets, skills line, summary)."""
    return _units(resume_text, split_sentences=True)
