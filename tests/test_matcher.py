import numpy as np

from resume_analyser.matcher import match_requirements

REQS = ["needs python", "needs kubernetes"]
BULLETS = ["wrote python services", "managed a retail store", "unrelated hobby"]


def fake_retrieve(requirements, bullets):
    # requirement 0 is most similar to bullet 0; requirement 1 to bullet 1 (a decoy)
    return np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.0]])


def make_rerank(table):
    def rerank(pairs):
        return [table.get(pair, 0.0) for pair in pairs]
    return rerank


def test_best_reranked_candidate_becomes_evidence():
    rerank = make_rerank({("needs python", "wrote python services"): 0.9})
    matches, coverage = match_requirements(REQS, BULLETS, fake_retrieve, rerank)
    assert matches[0].evidence == "wrote python services"
    assert matches[0].status == "met"
    assert matches[1].status == "missing"
    assert abs(coverage - 0.45) < 1e-9


def test_reranker_can_overrule_retrieval_order():
    # retrieval prefers bullet 1 for requirement 1, but the reranker finds bullet 2 better
    rerank = make_rerank({("needs kubernetes", "unrelated hobby"): 0.7})
    matches, _ = match_requirements(REQS, BULLETS, fake_retrieve, rerank, top_k=3)
    assert matches[1].evidence == "unrelated hobby"


def test_top_k_limits_reranker_work():
    seen = []

    def rerank(pairs):
        seen.extend(pairs)
        return [0.0] * len(pairs)

    match_requirements(REQS, BULLETS, fake_retrieve, rerank, top_k=1)
    assert len(seen) == len(REQS)  # one candidate per requirement


def test_partial_status_band():
    rerank = make_rerank({("needs python", "wrote python services"): 0.3})
    matches, _ = match_requirements(REQS, BULLETS, fake_retrieve, rerank)
    assert matches[0].status == "partial"


def test_empty_inputs():
    assert match_requirements([], BULLETS, fake_retrieve, make_rerank({})) == ([], 0.0)
    matches, coverage = match_requirements(REQS, [], fake_retrieve, make_rerank({}))
    assert coverage == 0.0 and all(m.status == "missing" and m.evidence == "" for m in matches)
