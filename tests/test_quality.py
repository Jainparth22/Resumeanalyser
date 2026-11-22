from resume_analyser.quality import QUANT_PATTERN, resume_quality_signals


def test_verb_forms_are_recognised():
    text = "Building APIs\nLeading a team\nDeployed models\nnothing here"
    assert resume_quality_signals(text)["action_verb_lines"] == 3


def test_years_and_phone_numbers_are_not_achievements():
    for noise in ["Graduated in 2023", "Phone: 9876543210", "Jan 2021 - Mar 2022"]:
        assert not QUANT_PATTERN.search(noise), noise


def test_real_metrics_are_counted():
    for metric in ["Improved latency by 30%", "Saved $2M", "Served 10k users", "3x throughput"]:
        assert QUANT_PATTERN.search(metric), metric


def test_missing_sections_reported():
    r = resume_quality_signals("Skills: Python\nProjects: demo")
    assert set(r["missing_sections"]) == {"experience", "education"}


def test_score_bounded_and_empty_safe():
    assert 0 <= resume_quality_signals("")["quality_score"] <= 100
