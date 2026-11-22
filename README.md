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
resume ─┐                          ┌─ semantic similarity (BGE, chunked, calibrated) ─ 45%
        ├─► analyze() ──► score ───┼─ skill coverage (Tier 1 taxonomy + Tier 2 KeyBERT) 35%
JD ─────┘                          └─ resume quality heuristics ──────────────────── 20%
```

| Stage | What it does |
|---|---|
| Semantic | `BAAI/bge-small-en-v1.5` embeddings; long texts are chunked and mean-pooled so nothing is lost to the 512-token limit; raw cosine is rescaled to a 0-100 score |
| Tier 1 (precision) | FlashText matching against 3,333 skills from the [MIND Tech Ontology](https://github.com/MIND-TechAI/MIND-tech-ontology) (MIT), with synonym handling and implied skills (Next.js → React → JavaScript). Ambiguous aliases (`go`, `do`, `ai`, `art`, ...) are filtered to avoid false positives |
| Tier 2 (recall) | KeyBERT keyphrases for anything the taxonomy doesn't know, matched with RapidFuzz + embedding similarity and clearly labelled *unverified* |
| Quality | Action verbs, quantified achievements (real metrics, not years/phone numbers), section presence |

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
| `EMBED_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Any compatible sentence-transformers model |

## Known limitations
- Scores are heuristic. The score weights and the cosine rescale bounds are hand-tuned, not learned from labelled data.
- The Tier 2 keyphrases are unverified and can be noisy; treat them as hints.
- English only. Inputs are plain text (no PDF/DOCX parsing yet).
- This tool is a writing aid, not a hiring decision system. Do not use the score to automatically reject candidates.

## Contributing
Issues and PRs welcome, see [CONTRIBUTING.md](CONTRIBUTING.md). Good first issues are listed there.

## License & credits
MIT, see [LICENSE](LICENSE). Skills data: [MIND Tech Ontology](https://github.com/MIND-TechAI/MIND-tech-ontology) (MIT). Embeddings: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5). Keyphrases: [KeyBERT](https://github.com/MaartenGr/KeyBERT).
