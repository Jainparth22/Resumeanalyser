"""AI Resume & Job Match Analyzer -- Gradio UI + public API.

All scoring logic lives in the `resume_analyser` package; this file is only
the web layer: validation, rate limiting and the Gradio interface.
"""

import os

import gradio as gr

from resume_analyser.engine import Engine
from resume_analyser.ratelimit import RateLimitExceeded, SlidingWindowRateLimiter

# -------------------------------
# Configuration (override with Space "Variables" / environment variables)
# -------------------------------
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "20000"))
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_WINDOW_SECONDS = float(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

# Models load once at startup (see resume_analyser/engine.py)
engine = Engine()
rate_limiter = SlidingWindowRateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)


def format_report(r):
    s, c, k, q = r["scores"], r["confirmed_skills"], r["other_keywords"], r["quality_signals"]
    join = lambda items: ", ".join(items) if items else "None"
    implied_note = (
        f" (of which via implied skills: {', '.join(c['matched_via_implication'])})"
        if c["matched_via_implication"]
        else ""
    )
    icon = {"met": "[OK]     ", "partial": "[PARTIAL]", "missing": "[MISSING]"}
    req_lines = "\n".join(
        f"{icon[m['status']]} {m['requirement']}"
        + (f"\n            evidence: {m['evidence']}" if m["status"] != "missing" else "")
        for m in r["requirements"]
    ) or "None detected"
    return f"""
1. OVERALL MATCH SCORE
{s['overall']}%  (semantic {s['semantic']}% . skill coverage {s['coverage']}% . requirement coverage {s['requirement_coverage']}% . resume quality {s['quality']}%)

2. JOB REQUIREMENTS -- EVIDENCE FROM YOUR RESUME (cross-encoder)
{req_lines}

3. CONFIRMED SKILLS -- MATCHED (vetted taxonomy){implied_note}
{join(c['matched'])}

4. CONFIRMED SKILLS -- MISSING (vetted taxonomy)
{join(c['missing'])}

5. OTHER KEYWORDS DETECTED IN RESUME (unverified, AI-extracted)
{join(k['matched'])}

6. OTHER KEYWORDS MISSING (unverified, AI-extracted -- double-check these manually)
{join(k['missing'])}

7. RESUME QUALITY SIGNALS
Action-verb bullet lines: {q['action_verb_lines']}
Quantified achievement lines: {q['quantified_lines']}
Missing sections: {join(q['missing_sections'])}

8. SUGGESTIONS
{chr(10).join(f'- {x}' for x in r['suggestions'])}
""".strip()


# -------------------------------
# Request handling: validation + rate limiting
# -------------------------------
def client_key(request):
    """Best-effort client identity. On HF Spaces the proxy sets X-Forwarded-For;
    outside a trusted proxy that header is client-controlled (see docs)."""
    if request is None:
        return "anonymous"
    forwarded = request.headers.get("x-forwarded-for") if request.headers else None
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "anonymous"


def handle_request(resume_text, job_description, request: gr.Request = None):
    if not resume_text or not resume_text.strip() or not job_description or not job_description.strip():
        raise gr.Error("Please provide both Resume and Job Description.")
    if len(resume_text) > MAX_INPUT_CHARS or len(job_description) > MAX_INPUT_CHARS:
        raise gr.Error(f"Input too long (max {MAX_INPUT_CHARS} characters each).")
    try:
        rate_limiter.check(client_key(request))
    except RateLimitExceeded as e:
        raise gr.Error(
            f"Rate limit exceeded ({RATE_LIMIT_REQUESTS} requests / {int(RATE_LIMIT_WINDOW_SECONDS)}s). "
            f"Try again in {e.retry_after:.0f}s."
        )
    result = engine.analyze(resume_text, job_description)
    return format_report(result), result


# -------------------------------
# Gradio UI (the same click handler is exposed as the public API: /analyze)
# -------------------------------
EXAMPLES = [
    [
        """Name: Ananya Verma

Summary:
Data Analyst with experience in data processing and visualization, looking to transition into machine learning roles.

Skills:
Python, SQL, Pandas, NumPy, Data Science, Scikit-learn, Git

Experience:
- Performed data cleaning and analysis using Python and SQL, improving report turnaround by 20%.
- Built regression and classification models using scikit-learn.
- Created dashboards and reports for business stakeholders.

Education:
M.Sc in Data Science
""",
        """Job Title: NLP / Machine Learning Engineer

Requirements:
- Strong experience in Python and Machine Learning
- Hands-on experience with NLP and Transformers
- Familiarity with Hugging Face libraries
- Experience with PyTorch
- Knowledge of SQL and data processing tools
- Experience deploying ML models is a plus
""",
    ],
    [
        "Experienced Python developer with strong background in machine learning, NLP, and data science.",
        "Looking for a data scientist skilled in Python, machine learning, NLP, SQL, and Hugging Face.",
    ],
]

with gr.Blocks(title="AI Resume & Job Match Analyzer") as demo:
    gr.Markdown(
        "# AI Resume & Job Match Analyzer\n"
        "Compares a resume with a job description using BGE embeddings, a vetted 3,300+ skill "
        "taxonomy (MIND Tech Ontology) for high-confidence matching, a cross-encoder that finds the "
        "resume line backing each job requirement, and KeyBERT as a gap-filler. Open source (MIT) -- free API usage: see the "
        "**Use via API** link in the footer."
    )
    with gr.Row():
        resume_in = gr.Textbox(lines=12, label="Resume Text")
        jd_in = gr.Textbox(lines=12, label="Job Description")
    run_btn = gr.Button("Analyze", variant="primary")
    report_out = gr.Textbox(lines=26, label="Analysis Result")
    with gr.Accordion("Structured JSON (same payload the API returns)", open=False):
        json_out = gr.JSON(label="JSON")
    gr.Examples(examples=EXAMPLES, inputs=[resume_in, jd_in])
    run_btn.click(
        handle_request,
        inputs=[resume_in, jd_in],
        outputs=[report_out, json_out],
        api_name="analyze",
    )

demo.queue(default_concurrency_limit=2, max_size=20)

if __name__ == "__main__":
    demo.launch()
