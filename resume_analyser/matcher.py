"""Requirement-level matching: retrieve candidate bullets, rerank, aggregate.

Pipeline per job description:
  1. retrieve  -- a bi-encoder similarity matrix [requirements x bullets] cheaply
                  narrows each requirement to its top-k candidate bullets;
  2. rerank    -- a cross-encoder reads (requirement, bullet) *together* and
                  scores each candidate pair (slower, more accurate);
  3. aggregate -- the best candidate is that requirement's evidence.

The two scorers are injected as plain callables so this module needs no ML
libraries and is unit-tested with fakes:

    retrieve(requirements, bullets) -> 2-D array-like of shape [R, B]
    rerank(pairs: list[(requirement, bullet)]) -> list[float] in 0..1
"""

from dataclasses import dataclass

MET_THRESHOLD = 0.45     # rerank score at/above which a requirement counts as met
PARTIAL_THRESHOLD = 0.2  # between this and MET_THRESHOLD = partially evidenced


@dataclass
class RequirementMatch:
    requirement: str
    evidence: str       # best-supporting resume line ("" if the resume has none)
    score: float        # 0..1 rerank score of that line
    status: str         # "met" | "partial" | "missing"

    def to_dict(self):
        return {
            "requirement": self.requirement,
            "evidence": self.evidence,
            "score": round(self.score, 4),
            "status": self.status,
        }


def _status(score, met, partial):
    if score >= met:
        return "met"
    if score >= partial:
        return "partial"
    return "missing"


def match_requirements(
    requirements, bullets, retrieve, rerank, top_k=3, met=MET_THRESHOLD, partial=PARTIAL_THRESHOLD
):
    """Return (matches, coverage) where coverage in [0,1] is the mean of each
    requirement's best rerank score (0 if there is nothing to compare)."""
    if not requirements:
        return [], 0.0
    if not bullets:
        return [RequirementMatch(r, "", 0.0, "missing") for r in requirements], 0.0

    sims = retrieve(requirements, bullets)
    k = min(top_k, len(bullets))

    pairs, owners = [], []
    for ri, requirement in enumerate(requirements):
        row = list(sims[ri])
        top = sorted(range(len(bullets)), key=lambda bi: row[bi], reverse=True)[:k]
        for bi in top:
            pairs.append((requirement, bullets[bi]))
            owners.append((ri, bi))

    scores = rerank(pairs)

    best = {}
    for (ri, bi), score in zip(owners, scores):
        if ri not in best or score > best[ri][0]:
            best[ri] = (float(score), bi)

    matches = []
    for ri, requirement in enumerate(requirements):
        score, bi = best[ri]
        matches.append(
            RequirementMatch(
                requirement=requirement,
                evidence=bullets[bi],
                score=score,
                status=_status(score, met, partial),
            )
        )
    coverage = sum(m.score for m in matches) / len(matches)
    return matches, coverage
