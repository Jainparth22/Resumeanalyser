# System & ML Design Notes

What exists today, what was deliberately left out, and a roadmap with the trade-offs. Items are tagged **[done]**, **[next]**, **[later]**.

## Current architecture

```
client ─► HF proxy ─► Gradio queue (max 20, concurrency 2)
                         └► handle_request: validate ─► rate limit (per client) ─► analyze()
                                                                      ├─ chunked BGE embeddings
                                                                      ├─ Tier 1 taxonomy (FlashText, in-memory)
                                                                      ├─ Tier 2 KeyBERT + RapidFuzz
                                                                      └─ quality heuristics
```
Stateless, single process, CPU-only, models loaded once at startup. Fine for a free Space (2 vCPU / 16 GB).

## System design

| Concern | Status | Notes |
|---|---|---|
| Input validation | **[done]** | Non-empty, ≤ 20k chars per field; bounds latency and memory |
| Rate limiting | **[done]** | Sliding window, per client, thread-safe, bounded key table. In-memory so single-replica only |
| Backpressure | **[done]** | Gradio queue `max_size=20`, `default_concurrency_limit=2` protect the CPU instead of thrashing |
| Stable API contract | **[done]** | Versioned JSON with additive-only evolution |
| Caching | **[next]** | `sha256(resume+jd+version)` → result (LRU, ~1k entries). Same pair resubmitted while editing is the common case; saves 100% of compute on hits. Also cache JD embeddings/keyphrases separately: one JD is scored against many resumes |
| Distributed rate limiting | **[later]** | Redis `INCR`+`EXPIRE` or token bucket behind the same `check(key)` interface when running > 1 replica |
| Observability | **[next]** | Structured logs (request id, latency per stage, input sizes, never the text), Prometheus counters/histograms (`/metrics`), alert on p95 latency and error rate |
| Privacy | **[next]** | Resumes are PII. Don't log inputs, state retention in README (currently: none stored), offer a self-host path, optionally redact emails/phones before processing |
| Async / batch API | **[later]** | `POST /jobs` → job id → poll/webhook, with a worker pool, for bulk scoring without holding connections |
| Auth & quotas | **[later]** | Optional API keys with per-key quotas so power users don't share the anonymous bucket |
| Deployment | **[later]** | Docker image, GitHub Actions: run `pytest` on PR, auto-push to the Space on `main`, pin model revisions |
| Resilience | **[later]** | Health endpoint, graceful degradation: if KeyBERT fails, return Tier 1 only (already partly true: failures return no keyphrases) |
| Scaling | **[later]** | Embeddings are the hot path: batch concurrent requests, ONNX/quantized model (2-4× faster on CPU), or move inference to a GPU endpoint |

## ML system design

| Concern | Status | Notes |
|---|---|---|
| Long-document handling | **[done]** | Chunk + mean-pool instead of silent 512-token truncation |
| Score calibration | **[done, heuristic]** | Linear rescale of cosine (0.45-0.85 → 0-100). Replace with a learned calibrator (isotonic / Platt) once labelled data exists |
| Precision-first skill matching | **[done]** | Taxonomy tier with ambiguity filtering and exact-case short terms; learned tier is explicitly labelled *unverified* |
| Evaluation set | **[next, highest value]** | 100-300 labelled (resume, JD, human match 1-5) pairs. Report Spearman correlation, skill-extraction precision/recall on hand-annotated resumes. Without this every other change is guesswork. Track in CI so regressions fail the build |
| Learned scoring weights | **[later]** | Replace hand weights `0.45/0.35/0.20` with logistic/ordinal regression on the eval set |
| Better retrieval model | **[later]** | Compare `bge-small` vs `bge-base`, `e5`, `gte`, and a cross-encoder reranker on JD-requirement ↔ resume-bullet pairs (more accurate than whole-document cosine) |
| Section-aware matching | **[later]** | Parse resume into sections; weight skills found in *Experience* above a bare *Skills* list, since claims with evidence are stronger |
| Explainability | **[next]** | Return the resume sentence that supports each matched skill and the closest-but-insufficient sentence for each missing one |
| Skill-level inference | **[later]** | Years / proficiency cues ("5 yrs", "led", "familiar with") to distinguish expert from beginner |
| Fairness & bias | **[next]** | Audit for proxies (names, schools, gaps, gendered words). Keep the tool advisory; never position it as an automated screener. Document this in a model card |
| Drift & monitoring | **[later]** | The taxonomy goes stale as tools appear. Log (privacy-safe) unrecognised high-frequency Tier 2 phrases → candidates to add to the taxonomy; re-run evals on every taxonomy/model update |
| Model/data versioning | **[next]** | Pin `EMBED_MODEL_NAME` revision, version the taxonomy file, include `version` in responses (done) so results are reproducible |
| Model card / dataset card | **[next]** | Publish the cleaned skills data and eval set as a Hugging Face Dataset with a card: intended use, limitations, license |

## Suggested order of work
1. Eval set + CI (unlocks everything else, and is a strong portfolio artefact).
2. Result cache + structured logging.
3. Evidence-based explanations.
4. Learned calibration and weights from the eval set.
5. Docker + GitHub Actions deploy, then Redis limiter if you ever run >1 replica.
