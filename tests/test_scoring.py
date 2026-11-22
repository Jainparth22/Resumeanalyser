import json

from resume_analyser.scoring import LegacyScorer, LinearScorer, calibrate_cosine, load_scorer

F = {"cosine": 0.65, "coverage": 0.5, "req_coverage": 0.4, "quality": 0.5}


def test_calibration_bounds():
    assert calibrate_cosine(0.0) == 0.0
    assert calibrate_cosine(1.0) == 100.0
    assert abs(calibrate_cosine(0.65) - 50.0) < 1e-9


def test_legacy_score_in_range():
    assert 0 <= LegacyScorer().score(F) <= 100


def test_linear_scorer_maps_rating_to_percent_and_clips():
    s = LinearScorer(intercept=1.0, coef={"req_coverage": 4.0})
    assert s.score({**F, "req_coverage": 0.0}) == 0.0
    assert s.score({**F, "req_coverage": 1.0}) == 100.0
    assert s.score({**F, "req_coverage": 5.0}) == 100.0  # clipped, never > 100


def test_load_scorer_falls_back_when_missing_or_corrupt(tmp_path):
    assert load_scorer(str(tmp_path / "nope.json")).name == "legacy"
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert load_scorer(str(bad)).name == "legacy"
    incomplete = tmp_path / "inc.json"
    incomplete.write_text(json.dumps({"coef": {}}))
    assert load_scorer(str(incomplete)).name == "legacy"


def test_load_scorer_reads_valid_file(tmp_path):
    good = tmp_path / "w.json"
    good.write_text(json.dumps({"intercept": 1.0, "coef": {"cosine": 2.0}, "meta": {"n": 1}}))
    scorer = load_scorer(str(good))
    assert scorer.name == "learned" and scorer.coef["cosine"] == 2.0


def test_shipped_weights_file_is_valid_and_sane():
    """Guards the committed resume_analyser/weights.json against corruption."""
    scorer = load_scorer()
    assert scorer.name == "learned"
    assert set(scorer.coef) == {"cosine", "coverage", "req_coverage", "quality"}
    strong = {"cosine": 0.85, "coverage": 1.0, "req_coverage": 0.8, "quality": 0.5}
    weak = {"cosine": 0.45, "coverage": 0.0, "req_coverage": 0.05, "quality": 0.5}
    assert scorer.score(strong) > scorer.score(weak)
    assert 0 <= scorer.score(weak) <= scorer.score(strong) <= 100
