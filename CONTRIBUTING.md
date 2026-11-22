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
| `resume_analyser/` | Model-free logic: taxonomy, quality, segmentation, requirement matcher, scoring, rate limiter. **Keep these free of torch/gradio imports** so tests stay fast. `engine.py` is the one place that loads models |
| `app.py` | Web layer only: validation, rate limiting, Gradio UI and API |
| `eval/` | Labelled pairs, ablation script, weight fitting (see `eval/README.md`) |
| `tests/` | pytest suites |
| `docs/` | API guide, system / ML design notes |

## Rules of thumb
- Every behaviour change needs a test. Found a false-positive skill? Add the term to `AMBIGUOUS_TERMS` **and** a parametrized case in `tests/test_taxonomy.py`.
- Changing features, thresholds, models or weights? Re-run `python eval/run_eval.py` and paste the before/after table in the PR. Features changed → `--recompute` then `--fit`.
- Keep the API response shape backwards compatible; additive fields are fine, renames need a version bump (`resume_analyser/__init__.py`).

## Good first issues
1. Find more ambiguous taxonomy aliases (run the analyzer on real-looking resumes) and add them with tests.
2. Add PDF / DOCX text extraction (`pypdf`, `python-docx`) as an optional input path.
3. **Improve the evaluation set**: independently re-rate `eval/labels.csv`, add harder pairs (near-misses, keyword-stuffed resumes, career changers) and report inter-rater agreement.
4. **Calibrate the Tier 2 threshold** (0.70) on data that isn't the eval set, or replace it with a learned matcher, then re-run `eval/run_eval.py --recompute --fit`.
5. Multilingual support via `bge-m3` or `multilingual-e5-small`.
6. Upstream: report/fix bad synonyms in the [MIND Tech Ontology](https://github.com/MIND-TechAI/MIND-tech-ontology).

## Pull requests
Small, focused PRs. Describe *what* and *why*, include test output. By contributing you agree your work is released under the MIT license.
