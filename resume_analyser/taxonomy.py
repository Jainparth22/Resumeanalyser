"""Tier 1: vetted skills taxonomy matching.

Data source: MIND Tech Ontology (MIT licensed)
https://github.com/MIND-TechAI/MIND-tech-ontology

Two real-world problems this module handles beyond a plain keyword lookup:

* Implied skills: "Next.js" implies "React" implies "JavaScript".
* Ambiguous terms: the ontology contains many very short or plain-English
  aliases ("go", "do", "ai", "art", "bit", "box", "got" ...). Matching them
  case-insensitively would credit a resume with skills it never mentions
  ("we decided to go ahead" -> Go). Such terms are either dropped
  (AMBIGUOUS_TERMS) or only matched with exact capitalisation (<= 2 chars).
"""

import json
import os
from functools import lru_cache

from flashtext import KeywordProcessor

TAXONOMY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "skills_taxonomy.json"
)

# Terms at or below this length are matched case-sensitively only
# ("C", "R", "Go", "JS" yes; "c", "r", "go", "js" in running text no).
SHORT_TERM_MAX_LEN = 2
MAX_IMPLIED_DEPTH = 5

# Aliases that are ordinary English words / acronyms too ambiguous to trust.
# Extend this set (and add a test) when you find a new false positive.
AMBIGUOUS_TERMS = frozenset(
    {
        # short / plain-English aliases
        "ai", "do", "got", "art", "bit", "box", "fan", "gas", "gin", "dub", "eel",
        "arm", "arc", "chi", "dot", "env", "es", "ge", "ga", "ft", "d", "_", "bs",
        "bb", "bc", "cb", "dg", "dp", "ds", "dx", "ee", "ef", "fab", "dav", "geo",
        "hey", "io", "ivy", "jet", "lit", "md", "mit", "moa", "nom", "np", "pd",
        "pg", "pt", "re", "roo", "rr", "sa", "sf", "sh", "sm", "tb", "ua", "vc",
        "vs", "wb", "wt", "xd", "zip", "ifs", "ini", "inv", "kt", "ky", "lc", "lt",
        "mw", "nok", "ols", "opa", "ora", "orb", "paw", "pig", "pl", "poi", "ps",
        "qs", "rn", "rx", "ses", "sip", "tox", "wiz", "ws", "zap", "cue", "duo",
        "dyn", "hs", "hf", "ft", "ml5", "q#",
        # lowercase library names that are everyday words
        "agenda", "arrow", "black", "bull", "click", "clap", "fire", "fabric",
        "frames", "lime", "locust", "mage", "needle", "rich", "thor", "tract",
        "viper", "expo", "invoke", "emails", "helmet", "shards", "regex", "repl",
        "caret", "chai", "pact", "rollup", "rhino", "moto", "cobra", "slog",
    }
)


class Taxonomy:
    """Skill matcher + implied-skill graph built from ontology entries."""

    def __init__(self, entries):
        self._implies = {}
        ci_terms = {}  # lowercased term -> canonical (case-insensitive match)
        cs_terms = {}  # exact-cased term -> canonical (case-sensitive match)

        candidates = []  # (is_alias, term, canonical)
        for entry in entries:
            canonical = entry["name"]
            self._implies[canonical] = set(entry.get("impliesKnowingSkills", []) or [])
            # Some entries list only variants (e.g. "Java" -> "JDK") and not the
            # plain word itself, so the canonical name is always matchable.
            candidates.append((False, canonical, canonical))
            for syn in entry.get("synonyms", []) or []:
                candidates.append((True, syn, canonical))
        # Canonical names are registered first so they win any alias collision.
        candidates.sort(key=lambda c: c[0])

        for _, term, canonical in candidates:
            term = term.strip()
            key = term.lower()
            if not term or not any(ch.isalnum() for ch in term):
                continue
            if key in AMBIGUOUS_TERMS:
                continue
            if len(key) <= SHORT_TERM_MAX_LEN:
                # Exact capitalisation only: the term as written if it already
                # contains capitals (canonical "Go", "C#"), and ALL CAPS ("JS").
                variants = {term.upper()}
                if term != key:
                    variants.add(term)
                for v in variants:
                    cs_terms.setdefault(v, canonical)
            else:
                ci_terms.setdefault(key, canonical)

        self._ci = KeywordProcessor(case_sensitive=False)
        for term, canonical in ci_terms.items():
            self._ci.add_keyword(term, canonical)
        self._cs = KeywordProcessor(case_sensitive=True)
        for term, canonical in cs_terms.items():
            self._cs.add_keyword(term, canonical)

        self.num_skills = len(self._implies)

    def confirmed_skills(self, text):
        """Canonical skill names actually present in `text` (high precision)."""
        if not text:
            return set()
        ci_hits = self._ci.extract_keywords(text, span_info=True)
        cs_hits = self._cs.extract_keywords(text, span_info=True)
        found = {name for name, _, _ in ci_hits}
        # A short case-sensitive hit ("C") must not fire inside a longer
        # case-insensitive match ("C++").
        for name, start, end in cs_hits:
            if not any(start < ce and cs_ < end for _, cs_, ce in ci_hits):
                found.add(name)
        return found

    def expand_with_implied(self, skills):
        """Add skills implied by `skills` (transitive, depth-capped, cycle-safe)."""
        expanded = set(skills)
        frontier = set(skills)
        for _ in range(MAX_IMPLIED_DEPTH):
            next_frontier = set()
            for skill in frontier:
                for implied in self._implies.get(skill, set()):
                    if implied not in expanded:
                        next_frontier.add(implied)
            if not next_frontier:
                break
            expanded |= next_frontier
            frontier = next_frontier
        return expanded


@lru_cache(maxsize=1)
def load_default_taxonomy():
    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        return Taxonomy(json.load(f))
