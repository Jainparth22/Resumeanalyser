"""Evaluate scorers against the labelled pairs and (optionally) fit weights.

    python eval/run_eval.py              # ablation table (reuses cached features)
    python eval/run_eval.py --recompute  # re-run the models, refresh the cache
    python eval/run_eval.py --fit        # also write resume_analyser/weights.json

Learned scorers are evaluated with leave-one-resume-out cross-validation: the
weights used to score a resume's pairs never saw that resume. Numbers on this
small, author-labelled set are *indicative only* (see eval/README.md).
"""

import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from resume_analyser.scoring import (  # noqa: E402
    FEATURE_NAMES,
    WEIGHTS_PATH,
    LegacyScorer,
    LinearScorer,
)

CACHE = os.path.join(HERE, "features.json")
RIDGE_LAMBDA = 1.0


def load_pairs():
    with open(os.path.join(HERE, "resumes.json"), encoding="utf-8") as f:
        resumes = json.load(f)
    with open(os.path.join(HERE, "jds.json"), encoding="utf-8") as f:
        jds = json.load(f)
    with open(os.path.join(HERE, "labels.csv"), newline="", encoding="utf-8") as f:
        labels = list(csv.DictReader(f))
    return resumes, jds, labels


def compute_features(recompute):
    resumes, jds, labels = load_pairs()
    if os.path.exists(CACHE) and not recompute:
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)

    from resume_analyser.engine import EMBED_MODEL_NAME, RERANKER_MODEL_NAME, Engine

    engine = Engine(use_keybert=True, scorer=LegacyScorer())
    rows = []
    for i, row in enumerate(labels, 1):
        feats, _ = engine.compute(resumes[row["resume_id"]], jds[row["jd_id"]])
        rows.append(
            {
                "resume_id": row["resume_id"],
                "jd_id": row["jd_id"],
                "rating": int(row["rating"]),
                "features": feats,
            }
        )
        print(f"  [{i}/{len(labels)}] {row['resume_id']} x {row['jd_id']}", file=sys.stderr)
    data = {"embed_model": EMBED_MODEL_NAME, "reranker_model": RERANKER_MODEL_NAME, "rows": rows}
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    return data


# ---- ridge regression (closed form, standardised, intercept unpenalised) ----
def fit_ridge(X, y, lam=RIDGE_LAMBDA):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    w_std = np.linalg.solve(A, Z.T @ (y - y.mean()))
    coef = w_std / sd
    intercept = float(y.mean() - mu @ coef)
    return intercept, coef


def feature_matrix(rows, names):
    return np.array([[r["features"][n] for n in names] for r in rows], dtype=float)


def cv_predictions(rows, names):
    """Leave-one-resume-out predictions of the rating."""
    y = np.array([r["rating"] for r in rows], dtype=float)
    X = feature_matrix(rows, names)
    groups = np.array([r["resume_id"] for r in rows])
    pred = np.zeros(len(rows))
    for g in set(groups):
        test = groups == g
        b0, coef = fit_ridge(X[~test], y[~test])
        pred[test] = b0 + X[test] @ coef
    return pred


# ---- metrics -----------------------------------------------------------------
def spearman(scores, ratings):
    value = spearmanr(scores, ratings).statistic
    return float(value) if value == value else 0.0


def pairwise_accuracy(scores, ratings):
    """Of all pairs with different human ratings, how often does the higher-rated
    one get the higher score? (ties in score count as half)"""
    right = total = 0.0
    for i, j in itertools.combinations(range(len(scores)), 2):
        if ratings[i] == ratings[j]:
            continue
        total += 1
        d = (scores[i] - scores[j]) * (ratings[i] - ratings[j])
        right += 1.0 if d > 0 else 0.5 if d == 0 else 0.0
    return right / total if total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--fit", action="store_true")
    args = ap.parse_args()

    data = compute_features(args.recompute)
    rows = data["rows"]
    ratings = [r["rating"] for r in rows]
    legacy = LegacyScorer()

    candidates = {
        "cosine only (raw)": [r["features"]["cosine"] for r in rows],
        "taxonomy+keyword coverage only": [r["features"]["coverage"] for r in rows],
        "cross-encoder requirement coverage only": [r["features"]["req_coverage"] for r in rows],
        "resume quality only (sanity: should be weak)": [r["features"]["quality"] for r in rows],
        "legacy blend (hand weights)": [legacy.score(r["features"]) for r in rows],
        "learned: cosine+coverage+quality (CV)": cv_predictions(rows, ("cosine", "coverage", "quality")),
        "learned: all four features (CV)": cv_predictions(rows, FEATURE_NAMES),
    }

    print(f"\n{len(rows)} labelled pairs | embed={data['embed_model']} | reranker={data['reranker_model']}\n")
    print(f"{'scorer':<48}{'Spearman':>10}{'Pairwise acc':>14}")
    print("-" * 72)
    for name, scores in candidates.items():
        print(f"{name:<48}{spearman(scores, ratings):>10.3f}{pairwise_accuracy(scores, ratings):>14.3f}")

    if args.fit:
        X = feature_matrix(rows, FEATURE_NAMES)
        y = np.array(ratings, dtype=float)
        b0, coef = fit_ridge(X, y)
        meta = {
            "trained_on": f"{len(rows)} author-labelled pairs",
            "ridge_lambda": RIDGE_LAMBDA,
            "cv_spearman": round(spearman(candidates["learned: all four features (CV)"], ratings), 4),
            "embed_model": data["embed_model"],
            "reranker_model": data["reranker_model"],
        }
        scorer = LinearScorer(b0, dict(zip(FEATURE_NAMES, coef)), meta)
        with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
            json.dump(scorer.to_dict(), f, indent=2)
        print(f"\nwrote {WEIGHTS_PATH}")
        print("coefficients:", {k: round(v, 3) for k, v in scorer.coef.items()}, "intercept", round(b0, 3))


if __name__ == "__main__":
    main()
