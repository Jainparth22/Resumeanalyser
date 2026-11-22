"""Turn per-pair features into the final 0-100 match score.

Two scorers share one interface (`.score(features) -> 0..100`):

* LegacyScorer -- the original hand-weighted blend (kept as the baseline and
                  as the fallback when no learned weights file is present).
* LinearScorer -- ridge-regression weights learned on the labelled evaluation
                  set by eval/fit_weights.py and stored in weights.json.

Features (all produced by the engine):
    cosine        raw bi-encoder cosine similarity of resume vs JD (-1..1)
    coverage      share of JD skills/keywords found in the resume (0..1)
    req_coverage  mean best cross-encoder score over JD requirements (0..1)
    quality       resume writing-quality heuristic (0..1)
"""

import json
import os

FEATURE_NAMES = ("cosine", "coverage", "req_coverage", "quality")
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.json")

COSINE_FLOOR = 0.45
COSINE_CEIL = 0.85


def calibrate_cosine(cosine, floor=COSINE_FLOOR, ceil=COSINE_CEIL):
    """Heuristic linear rescale of raw cosine to 0..100."""
    scaled = (cosine - floor) / (ceil - floor)
    return min(max(scaled, 0.0), 1.0) * 100


class LegacyScorer:
    name = "legacy"

    def score(self, f):
        return round(
            0.45 * calibrate_cosine(f["cosine"]) + 0.35 * f["coverage"] * 100 + 0.20 * f["quality"] * 100,
            2,
        )


class LinearScorer:
    """pred_rating = intercept + sum(coef[name] * feature[name]);
    rating 1..5 is mapped linearly onto 0..100."""

    name = "learned"

    def __init__(self, intercept, coef, meta=None):
        self.intercept = float(intercept)
        self.coef = {k: float(v) for k, v in coef.items()}
        self.meta = meta or {}

    def predict_rating(self, f):
        return self.intercept + sum(self.coef.get(n, 0.0) * f[n] for n in FEATURE_NAMES)

    def score(self, f):
        pct = (self.predict_rating(f) - 1.0) / 4.0 * 100
        return round(min(max(pct, 0.0), 100.0), 2)

    def to_dict(self):
        return {"intercept": self.intercept, "coef": self.coef, "meta": self.meta}


def load_scorer(path=WEIGHTS_PATH):
    """Learned scorer if a valid weights file exists, else the legacy baseline."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return LinearScorer(data["intercept"], data["coef"], data.get("meta"))
    except (OSError, KeyError, ValueError, TypeError):
        return LegacyScorer()
