"""ML engine: loads the models once and turns (resume, JD) into features,
a score and an explainable result. Used by app.py and eval/run_eval.py.
"""

import math
import os

import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import CrossEncoder, SentenceTransformer, util

from . import __version__
from .matcher import match_requirements
from .quality import resume_quality_signals
from .scoring import calibrate_cosine, load_scorer
from .segmentation import split_bullets, split_requirements
from .taxonomy import load_default_taxonomy

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/stsb-distilroberta-base")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CHUNK_CHARS = 1200  # ~300 tokens, safely under BGE's 512-token window
FUZZY_MATCH_THRESHOLD = 85
# BGE cosines are compressed (unrelated phrases already score ~0.6), so 0.55 matched everything.
# 0.70 chosen from same-role vs different-role phrase similarities on the eval set (heuristic).
SEMANTIC_MATCH_THRESHOLD = 0.70
RETRIEVE_TOP_K = 3

# Generic job-posting / resume filler that KeyBERT otherwise surfaces as "keywords".
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # noqa: E402

KEYPHRASE_STOP_WORDS = sorted(
    set(ENGLISH_STOP_WORDS)
    | {
        "job", "title", "requirement", "requirements", "responsibilities", "experience", "experienced",
        "strong", "solid", "knowledge", "skills", "skill", "ability", "familiarity", "hands", "proficiency",
        "understanding", "focus", "plus", "looking", "work", "working", "team", "candidate", "role",
        "summary", "education", "projects", "using", "built", "created", "performed",
    }
)


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


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


class Engine:
    def __init__(self, use_keybert=True, scorer=None):
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)
        self.kw_model = None
        if use_keybert:
            from keybert import KeyBERT

            self.kw_model = KeyBERT(model=self.model)  # reuses the same embedding model
        self.taxonomy = load_default_taxonomy()
        self.scorer = scorer or load_scorer()

    # ---- embeddings -------------------------------------------------------
    def _encode(self, texts, is_query=False, **kwargs):
        if is_query:
            texts = [BGE_QUERY_PREFIX + t for t in texts] if not isinstance(texts, str) else BGE_QUERY_PREFIX + texts
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False, **kwargs)

    def embed_long(self, text, is_query=False):
        """Mean-pool chunk embeddings into one unit vector for the whole document."""
        vecs = self._encode(split_chunks(text), is_query=is_query)
        mean = np.asarray(vecs).mean(axis=0)
        return mean / (np.linalg.norm(mean) or 1.0)

    # ---- requirement-level matching --------------------------------------
    def _retrieve(self, requirements, bullets):
        req = self._encode(requirements, is_query=True)
        bul = self._encode(bullets)
        return np.asarray(req) @ np.asarray(bul).T

    def _rerank(self, pairs):
        scores = np.asarray(self.reranker.predict(pairs, batch_size=32, show_progress_bar=False), dtype=float)
        if scores.size and (scores.min() < -1.0 or scores.max() > 1.5):  # clearly raw logits -> probabilities
            scores = np.array([_sigmoid(x) for x in scores])
        return np.clip(scores, 0.0, 1.0).tolist()  # STS models can overshoot 0..1 slightly

    def requirement_matches(self, resume_text, job_description):
        return match_requirements(
            split_requirements(job_description),
            split_bullets(resume_text),
            self._retrieve,
            self._rerank,
            top_k=RETRIEVE_TOP_K,
        )

    # ---- tier 2 gap-filler -----------------------------------------------
    def extract_keyphrases(self, text, top_n=15):
        if self.kw_model is None or not text or not text.strip():
            return []
        try:
            raw = self.kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words=KEYPHRASE_STOP_WORDS,
                top_n=top_n,
                use_mmr=True,
                diversity=0.6,
            )
        except Exception:
            return []
        return [p.strip().lower() for p, _ in raw if len(p.strip()) >= 2]

    def tier2_unverified_keywords(self, jd_keyphrases, resume_text, resume_keyphrases, already_confirmed_jd):
        """Only JD keyphrases Tier 1 did not recognise, so Tier 2 never double-counts."""
        confirmed_lower = {s.lower() for s in already_confirmed_jd}
        candidates = [
            p for p in jd_keyphrases
            if not any(fuzz.ratio(p, c) > FUZZY_MATCH_THRESHOLD for c in confirmed_lower)
        ]
        if not candidates:
            return [], []

        resume_norm = resume_text.lower().replace("-", " ").replace(".", "")
        jd_emb = self._encode(candidates, convert_to_tensor=True)
        res_emb = self._encode(resume_keyphrases, convert_to_tensor=True) if resume_keyphrases else None

        matched, missing = [], []
        for i, phrase in enumerate(candidates):
            norm = phrase.replace("-", " ").replace(".", "")
            if fuzz.partial_ratio(norm, resume_norm) >= FUZZY_MATCH_THRESHOLD:
                matched.append(phrase)
            elif res_emb is not None and util.cos_sim(jd_emb[i], res_emb)[0].max().item() >= SEMANTIC_MATCH_THRESHOLD:
                matched.append(phrase)
            else:
                missing.append(phrase)
        return matched, missing

    # ---- features + final result -----------------------------------------
    def compute(self, resume_text, job_description):
        """All model work. Returns (features, details); features feed the scorer."""
        cosine = float(np.dot(self.embed_long(resume_text), self.embed_long(job_description, is_query=True)))

        tax = self.taxonomy
        jd_confirmed = tax.confirmed_skills(job_description)
        resume_raw = tax.confirmed_skills(resume_text)
        resume_effective = tax.expand_with_implied(resume_raw)
        matched_confirmed = jd_confirmed & resume_effective
        missing_confirmed = jd_confirmed - resume_effective

        jd_body = "\n".join(split_requirements(job_description)) or job_description
        jd_kp = self.extract_keyphrases(jd_body, top_n=15)
        res_kp = self.extract_keyphrases(resume_text, top_n=20)
        t2_matched, t2_missing = self.tier2_unverified_keywords(jd_kp, resume_text, res_kp, jd_confirmed)

        total = len(jd_confirmed) + len(t2_matched) + len(t2_missing)
        coverage = (len(matched_confirmed) + len(t2_matched)) / total if total else 1.0

        matches, req_coverage = self.requirement_matches(resume_text, job_description)
        quality = resume_quality_signals(resume_text)

        features = {
            "cosine": cosine,
            "coverage": coverage,
            "req_coverage": req_coverage,
            "quality": quality["quality_score"] / 100.0,
        }
        details = {
            "jd_confirmed": jd_confirmed,
            "matched_confirmed": matched_confirmed,
            "missing_confirmed": missing_confirmed,
            "implied_only": {s for s in matched_confirmed if s not in resume_raw},
            "t2_matched": t2_matched,
            "t2_missing": t2_missing,
            "matches": matches,
            "quality": quality,
        }
        return features, details

    def analyze(self, resume_text, job_description):
        f, d = self.compute(resume_text, job_description)
        q = d["quality"]
        overall = self.scorer.score(f)

        suggestions = []
        unmet = [m for m in d["matches"] if m.status == "missing"]
        if unmet:
            suggestions.append(
                "No clear evidence in your resume for: " + "; ".join(f'"{m.requirement}"' for m in unmet[:3]) + "."
            )
        if d["missing_confirmed"]:
            suggestions.append(
                f"Confirmed JD skills missing from your resume: {', '.join(sorted(d['missing_confirmed'])[:8])}."
            )
        if d["t2_missing"]:
            suggestions.append(
                f"Also worth checking (auto-detected, not taxonomy-verified): {', '.join(d['t2_missing'][:5])}."
            )
        if q["missing_sections"]:
            suggestions.append(
                f"Resume appears to be missing a clear '{', '.join(q['missing_sections'])}' section -- "
                "ATS parsers rely on section headers to categorize content."
            )
        if q["quantified_lines"] == 0:
            suggestions.append(
                "No quantified achievements detected (numbers, %, $, scale). Try rephrasing bullets like "
                "'Improved pipeline performance' as 'Improved pipeline throughput by 30%'."
            )
        if not suggestions:
            suggestions.append("Strong match -- resume aligns well with this job description.")

        return {
            "version": __version__,
            "scores": {
                "overall": overall,
                "scorer": self.scorer.name,
                "semantic": round(calibrate_cosine(f["cosine"]), 2),
                "semantic_raw_cosine": round(f["cosine"], 4),
                "coverage": round(f["coverage"] * 100, 2),
                "requirement_coverage": round(f["req_coverage"] * 100, 2),
                "quality": round(f["quality"] * 100, 2),
            },
            "requirements": [m.to_dict() for m in d["matches"]],
            "confirmed_skills": {
                "matched": sorted(d["matched_confirmed"]),
                "missing": sorted(d["missing_confirmed"]),
                "matched_via_implication": sorted(d["implied_only"]),
            },
            "other_keywords": {"matched": d["t2_matched"], "missing": d["t2_missing"]},
            "quality_signals": {
                "action_verb_lines": q["action_verb_lines"],
                "quantified_lines": q["quantified_lines"],
                "missing_sections": q["missing_sections"],
            },
            "suggestions": suggestions,
        }
