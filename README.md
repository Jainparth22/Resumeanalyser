---
title: ResumeAnalyser
emoji: 👁
colorFrom: gray
colorTo: purple
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: mit
short_description: Resume-to-JD match scoring with a free API
tags:
  - resume
  - nlp
  - sentence-transformers
  - keybert
  - gradio
  - open-source
---

# AI Resume & Job Match Analyzer

Open-source (MIT) resume-vs-job-description analyzer. Paste a resume and a job description and get a match score, the skills you have and are missing, resume-quality signals and concrete suggestions. It runs as a web UI **and** as a free API you can call from your own project.

Originally built for the Hugging Face x IITM workshop and now maintained as an open-source project.

## How it works

```
resume ─┐                           ┌─ semantic similarity (BGE, chunked) ──────────┐
        ├─► analyze() ─► features ──┼─ skill coverage (taxonomy + KeyBERT) ─────────┼─► scorer ─► 0-100
JD ─────┘        │                  ├─ requirement coverage (retrieve → rerank) ────┤
                 │                  └─ resume quality heuristics ───────────────────┘
                 └─► per-requirement evidence: the resume line that backs each JD requirement
```

| Stage | What it does |
|---|---|
| Semantic | `BAAI/bge-small-en-v1.5` embeddings; long texts are chunked and mean-pooled so nothing is lost to the 512-token limit |
| Requirement matching | The JD is split into requirements and the resume into lines. The bi-encoder retrieves the top-3 candidate lines per requirement; a cross-encoder (`stsb-distilroberta-base`) re-scores them; the best line is returned as **evidence** with a met / partial / missing status |
| Tier 1 (precision) | FlashText matching against 3,333 skills from the [MIND Tech Ontology](https://github.com/MIND-TechAI/MIND-tech-ontology) (MIT), with synonym handling and implied skills (Next.js → React → JavaScript). Ambiguous aliases (`go`, `do`, `ai`, `art`, ...) are filtered to avoid false positives |
| Tier 2 (recall) | KeyBERT keyphrases for anything the taxonomy doesn't know, matched with RapidFuzz + embedding similarity (threshold 0.70; BGE cosines are compressed, the original 0.55 matched everything) and clearly labelled *unverified* |
| Quality | Action verbs, quantified achievements (real metrics, not years/phone numbers), section presence |
| Scorer | Blends the four features. Hand-weighted by default; learned weights are used automatically if `resume_analyser/weights.json` exists (see [eval/](eval/README.md)) |

## Evaluation

54 resume/JD pairs (9 resumes × 6 roles) rated 1-5 for fit. Spearman correlation with the rating; learned rows use leave-one-resume-out cross-validation:

| Scorer | Spearman | Pairwise acc. |
|---|---|---|
| Resume quality only (sanity check, should be weak) | 0.276 | 0.618 |
| Raw cosine similarity only | 0.876 | 0.929 |
| Requirement coverage only (retrieve → rerank) | 0.872 | 0.927 |
| Taxonomy + keyword coverage only | 0.887 | 0.938 |
| Hand-weighted blend (previous default) | 0.874 | 0.932 |
| Learned, no requirement feature (CV) | 0.908 | 0.952 |
| **Learned, all features (CV), current default** | **0.909** | **0.954** |

**How much weight to give this:** the labels are first-pass ratings written by the project author's AI assistant, not independent human judgement, and the pairs are easy (clearly different role families). The differences between the top rows are small and well inside what 54 pairs can distinguish. What the table does support: the quality score is a poor predictor of fit (so the learned weights drop it), and every content-based signal lands in a similar band; a bug-fix to Tier 2 (see below) moved keyword coverage from 0.66 to 0.89. What it does not support: any claim of production-grade accuracy. The main value of the requirement matcher is the per-requirement **evidence**, not the headline score. Reproduce with `python eval/run_eval.py`; details and caveats in [eval/README.md](eval/README.md).

## Use it

### 1. Web UI
Open the Space and paste text. The *Structured JSON* panel shows exactly what the API returns.

### 2. API (Python)
```bash
pip install gradio_client
```
```python
from gradio_client import Client

client = Client("YOUR_HF_USERNAME/ResumeAnalyser")   # <- the Space id
report_text, result = client.predict(
    "Python developer, built NLP models with PyTorch, improved accuracy by 12%.",
    "We need an ML engineer with Python, PyTorch and NLP experience.",
    api_name="/analyze",
)
print(result["scores"]["overall"], result["confirmed_skills"]["missing"])
```

### 3. API (curl / any language)
```bash
curl -s -X POST https://YOUR_HF_USERNAME-resumeanalyser.hf.space/gradio_api/call/analyze \
  -H "Content-Type: application/json" \
  -d '{"data": ["<resume text>", "<job description>"]}'
# -> {"event_id": "abc123"}
curl -N https://YOUR_HF_USERNAME-resumeanalyser.hf.space/gradio_api/call/analyze/abc123
```

Full request/response schema, limits, error handling and integration patterns: **[docs/API.md](docs/API.md)**.

### 4. Run locally
```bash
git clone https://github.com/Jainparth22/Resumeanalyser && cd Resumeanalyser
pip install -r requirements-dev.txt
python app.py          # http://127.0.0.1:7860
pytest                 # model-free unit tests, runs in <1s
python eval/run_eval.py   # ablation on the labelled pairs
```

### 5. Reuse the core library
The model-free parts import without torch or gradio:
```python
from resume_analyser.taxonomy import load_default_taxonomy
from resume_analyser.quality import resume_quality_signals

tax = load_default_taxonomy()
skills = tax.confirmed_skills("Built services in Go with Next.js")
print(tax.expand_with_implied(skills))   # includes React, JavaScript, ...
```

## Limits & configuration

Default limits: **10 requests / 60 s per client**, **20,000 characters** per field. Override with Space variables (Settings → Variables) or environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `RATE_LIMIT_REQUESTS` | `10` | Requests allowed per window per client |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Window length |
| `MAX_INPUT_CHARS` | `20000` | Max characters per input field |
| `EMBED_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Bi-encoder for similarity and retrieval |
| `RERANKER_MODEL_NAME` | `cross-encoder/stsb-distilroberta-base` | Cross-encoder that scores requirement/evidence pairs |

## Known limitations
- Scores are heuristic. See *Evaluation* for how they were checked and how small and author-labelled that check is.
- Tier 2 keyphrases are unverified hints. Their similarity threshold (0.70) was chosen on the same small eval set, so it may not transfer.
- Requirement/evidence thresholds (met ≥ 0.45, partial ≥ 0.2) were set by inspecting a handful of pairs, not tuned.
- English only. Inputs are plain text (no PDF/DOCX parsing yet).
- This tool is a writing aid, not a hiring decision system. Do not use the score to automatically reject candidates.

## Contributing
Issues and PRs welcome, see [CONTRIBUTING.md](CONTRIBUTING.md). Good first issues are listed there.

## License & credits
MIT, see [LICENSE](LICENSE). Skills data: [MIND Tech Ontology](https://github.com/MIND-TechAI/MIND-tech-ontology) (MIT). Embeddings: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5). Keyphrases: [KeyBERT](https://github.com/MaartenGr/KeyBERT).
