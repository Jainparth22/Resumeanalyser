import pytest

from resume_analyser.taxonomy import Taxonomy, load_default_taxonomy


@pytest.fixture(scope="module")
def tax():
    return load_default_taxonomy()


def test_loads_full_ontology(tax):
    assert tax.num_skills > 3000


def test_synonym_maps_to_canonical(tax):
    assert "Kubernetes" in tax.confirmed_skills("deployed on k8s")


@pytest.mark.parametrize(
    "text",
    [
        "We decided to go ahead and do it",
        "a bit of art in the box",
        "I got an AI-related degree",
    ],
)
def test_common_words_are_not_skills(tax, text):
    assert tax.confirmed_skills(text) == set()


def test_short_skills_need_exact_case(tax):
    assert "Go" in tax.confirmed_skills("Backend services in Go and Python")
    assert "R" in tax.confirmed_skills("Statistics with R")
    assert "Go" not in tax.confirmed_skills("let's go to market")


def test_cplusplus_does_not_also_credit_c(tax):
    found = tax.confirmed_skills("Systems work in C++")
    assert "C++" in found and "C" not in found


def test_implied_skills_are_transitive_and_include_origin(tax):
    expanded = tax.expand_with_implied({"Next.js"})
    assert {"Next.js", "React", "JavaScript"} <= expanded


def test_implied_expansion_survives_cycles():
    t = Taxonomy(
        [
            {"name": "Alpha", "synonyms": [], "impliesKnowingSkills": ["Beta"]},
            {"name": "Beta", "synonyms": [], "impliesKnowingSkills": ["Alpha"]},
        ]
    )
    assert t.expand_with_implied({"Alpha"}) == {"Alpha", "Beta"}


def test_canonical_name_wins_alias_collision():
    t = Taxonomy(
        [
            {"name": "Thing One", "synonyms": ["widget"], "impliesKnowingSkills": []},
            {"name": "widget", "synonyms": [], "impliesKnowingSkills": []},
        ]
    )
    assert t.confirmed_skills("I use widget daily") == {"widget"}


def test_empty_text(tax):
    assert tax.confirmed_skills("") == set()
