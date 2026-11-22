from resume_analyser.segmentation import MAX_UNITS, split_bullets, split_requirements

JD = """Job Title: ML Engineer

Requirements:
- Strong experience in Python and PyTorch
* Hands-on experience with NLP
1. Experience deploying ML models to production
- SQL
"""


def test_requirements_strip_markers_headings_and_title():
    reqs = split_requirements(JD)
    assert reqs == [
        "Strong experience in Python and PyTorch",
        "Hands-on experience with NLP",
        "Experience deploying ML models to production",
    ]  # "Job Title", "Requirements:" and the 1-word "SQL" line are not requirements


def test_paragraph_is_split_into_sentences():
    reqs = split_requirements("We need Python skills. You should know SQL well. Also Docker experience.")
    assert len(reqs) == 3


def test_resume_bullets():
    resume = "Experience:\n- Built APIs with FastAPI serving 2M requests\n- Led a team of five engineers\nSkills: Python, SQL, Docker"
    bullets = split_bullets(resume)
    assert "Built APIs with FastAPI serving 2M requests" in bullets
    assert "Skills: Python, SQL, Docker" in bullets
    assert "Experience:" not in bullets


def test_unit_count_is_capped_and_empty_is_safe():
    assert len(split_bullets("\n".join(f"- did task number {i} well" for i in range(200)))) == MAX_UNITS
    assert split_requirements("") == [] and split_bullets("   \n ") == []
