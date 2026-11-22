# Contributing

Thanks for helping! This project is small on purpose: a PR can land in an evening.

## Setup
```bash
pip install -r requirements-dev.txt
pytest            # model-free tests, <1s
python app.py     # full app (downloads the embedding model on first run)
```

## Layout
| Path | What |
|---|---|
| `resume_analyser/` | Model-free logic: taxonomy matching, quality heuristics, rate limiter. **Keep this free of torch/gradio imports** so it stays fast to test |
| `app.py` | Models, scoring pipeline, Gradio UI and API |
| `tests/` | pytest suites |
| `docs/` | API guide, system / ML design notes |

## Rules of thumb
- Every behaviour change needs a test. Found a false-positive skill? Add the term to `AMBIGUOUS_TERMS` **and** a parametrized case in `tests/test_taxonomy.py`.
- Changing score weights or the cosine calibration? Explain why in the PR (ideally with before/after on sample pairs).
- Keep the API response shape backwards compatible; additive fields are fine, renames need a version bump (`resume_analyser/__init__.py`).

## Good first issues
1. Find more ambiguous taxonomy aliases (run the analyzer on real-looking resumes) and add them with tests.
2. Add PDF / DOCX text extraction (`pypdf`, `python-docx`) as an optional input path.
3. Add a labelled evaluation set (`eval/pairs.jsonl`) and a script that reports score vs human rating correlation.
4. Per-skill evidence: return the resume sentence that matched each skill.
5. Multilingual support via `bge-m3` or `multilingual-e5-small`.
6. Upstream: report/fix bad synonyms in the [MIND Tech Ontology](https://github.com/MIND-TechAI/MIND-tech-ontology).

## Pull requests
Small, focused PRs. Describe *what* and *why*, include test output. By contributing you agree your work is released under the MIT license.
