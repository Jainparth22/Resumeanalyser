# Putting this on your resume (honestly) and defending it

This file is for the maintainer, not for users. It is meant to be deleted or kept private if you prefer.

## The honest framing

- **True:** you designed, built, tested, documented and deployed an open-source (MIT) project on Hugging Face Spaces, with a public API other people can call.
- **Not true:** that you contributed *to Hugging Face*. Hugging Face hosts it; it is your project. Interviewers who click the link will see this immediately, so don't claim otherwise. The real, defensible claim is stronger than a vague one.
- If you want a literal "contributor to a Hugging Face repo" line, make a merged PR to `gradio`, `sentence-transformers`, `transformers` or `hub-docs` (see CONTRIBUTING.md → Good first issues for ideas), then list it separately.

## Resume bullets (pick 2-3, keep numbers you can verify)

**Project entry**
> **AI Resume & Job Match Analyzer** — open-source (MIT), live on Hugging Face Spaces | Python, Gradio, sentence-transformers, KeyBERT
> - Built a hybrid matching pipeline: BGE embeddings (chunked + calibrated) for semantic similarity, a 3,300-skill ontology with synonym and implied-skill expansion for precise skill matching, and KeyBERT as a recall gap-filler.
> - Cut false-positive skill matches by filtering ambiguous aliases (e.g. "go", "ai", "art") and enforcing case-sensitive matching for short terms; covered with a pytest suite (20 tests) that runs without ML models.
> - Shipped a documented public API (Python/JS/curl) with per-client sliding-window rate limiting, input validation and queue backpressure.
> - Wrote system/ML design docs and contribution guide; published under MIT with attribution to upstream data.

Add real numbers once you have them: Space likes/visits, number of API users or forks, GitHub stars, eval correlation (e.g. "Spearman ρ = 0.6 against 150 human-rated pairs"). **Don't invent metrics.** An empty number is better than an unverifiable one.

**Journey / profile line**
> Open-source maintainer: built and published an NLP resume-matching tool on Hugging Face (workshop project → documented, tested, API-enabled, MIT-licensed).

## How to defend it in an interview

**"Walk me through it."** Two-tier idea: a precise tier you can trust (taxonomy) plus a flexible tier you label as uncertain (KeyBERT); combine with semantic similarity and writing-quality signals. Precision where it matters, recall where it's cheap.

**"Why not just use embeddings / an LLM?"** Embeddings give a fuzzy topical score but can't tell you *which* skill is missing. An LLM is costlier, slower, non-deterministic and harder to test. The taxonomy is deterministic, explainable and free. The embedding captures what the taxonomy can't.

**"How do you know it works?"** Be candid: *"Right now I validate with unit tests and manual cases; the scoring weights and calibration bounds are heuristic. The next step in my roadmap is a labelled evaluation set so I can measure correlation and tune weights."* Saying what you haven't done and what you'd do next reads as maturity; overclaiming reads as the opposite. Then point to `docs/SYSTEM_DESIGN.md`.

**"What was the hardest bug?"** Taxonomy false positives: words like "go" or "do" being counted as skills, and `C++` also crediting `C`. Fixed with an ambiguity list, case-sensitive short terms and span de-duplication, each with a regression test.

**"What about the 512-token limit?"** Whole resumes exceed it; the first version silently truncated. Now chunk, embed, mean-pool.

**"Scaling / rate limiting?"** Sliding-window limiter, per client, bounded memory, in-memory so it's single-replica. For multiple replicas: Redis-backed limiter behind the same interface. Queue max size and concurrency protect CPU.

**"Ethics?"** Resumes are PII and automated screening can encode bias. It is advisory, stores nothing, and the README says not to use the score as an automatic rejector.

**"What's your contribution vs. libraries?"** Own it precisely: the library choices are standard; *you* built the tiering, the ambiguity handling, implied-skill scoring, calibration, chunking, rate limiter, API contract, tests and docs. Don't claim you built BGE, KeyBERT or the ontology.

## Before you publish the claim: checklist
- [ ] Space is public and loads; README shows license MIT
- [ ] `YOUR_HF_USERNAME` placeholders replaced with the real Space id
- [ ] GitHub repo and Space are in sync; `pytest` passes
- [ ] You can explain every file in the repo without reading it
- [ ] Optional but strong: eval set + a one-paragraph results section in the README
