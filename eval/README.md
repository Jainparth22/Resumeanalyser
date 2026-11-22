# Evaluation

How we check whether the match score tracks human judgement of fit.

## Files
| File | Purpose |
|---|---|
| `build_dataset.py` | Source of truth: 9 resumes × 6 job descriptions = 54 rated pairs. Regenerates the three files below |
| `resumes.json`, `jds.json`, `labels.csv` | Generated data (rating 1-5: 5 strong fit … 1 unrelated) |
| `features.json` | Cached per-pair model features, so ablations and fitting don't re-run the models |
| `run_eval.py` | Ablation table (Spearman, pairwise accuracy); `--fit` writes `resume_analyser/weights.json` |

```bash
python eval/run_eval.py --recompute   # run models, refresh features.json, print the table
python eval/run_eval.py               # table from cache
python eval/run_eval.py --fit         # also fit and save learned weights
```

## Read this before quoting any number
- **The labels are not independent human ratings.** They were written once, by the same assistant that helped build the system, to bootstrap the evaluation. Re-rate them yourself (ideally two people, report agreement) before treating results as evidence.
- **The set is small and easy.** 54 pairs across 6 clearly separated role families. It catches gross errors ("retail manager scores like an ML engineer") but not fine distinctions such as senior vs mid-level or adjacent stacks. Add hard pairs (near-misses, keyword-stuffed resumes, career changers).
- **Learned scorers are cross-validated** with leave-one-resume-out, so the weights scoring a resume never saw it. The `--fit` weights saved to disk are trained on all 54 pairs, and so are optimistic if you re-score the same pairs.
- Spearman = rank correlation with the human rating (1 = perfect). Pairwise accuracy = share of differently-rated pairs ordered correctly.

## Adding data
Edit `RESUMES`, `JDS`, `RATINGS` in `build_dataset.py`, run it, then `python eval/run_eval.py --recompute`. Do not commit real candidates' resumes: use synthetic or properly anonymised, consented text.
