"""
Centralised prompt templates for every LLM-powered service.
Each prompt enforces strict JSON output as per §9 of the spec.
"""

# ─────────────────────────────────────────────────────────────────
# Document Classifier
# ─────────────────────────────────────────────────────────────────
DOCUMENT_CLASSIFIER = """\
You are a document classification agent for an investment research system.
Given the first 2000 characters of a document, classify it and extract metadata.

Respond ONLY with a JSON object. No preamble, no markdown fences.

Schema:
{{
  "document_type": "earnings_release" | "transcript" | "presentation" | "10-Q" | "10-K" | "annual_report" | "investor_letter" | "other",
  "company_ticker": "<ticker or null>",
  "period_label": "<e.g. 2026_Q1 or null>",
  "title": "<best guess title>",
  "language": "<ISO 639-1 code>",
  "confidence": <0.0-1.0>
}}

--- DOCUMENT TEXT (first 2000 chars) ---
{text}
"""

# ─────────────────────────────────────────────────────────────────
# KPI Extractor
# ─────────────────────────────────────────────────────────────────
KPI_EXTRACTOR = """\
You are a KPI extraction agent for an investment research system.
Extract ONLY explicitly stated quantitative metrics from the document text.

RULES:
- Do NOT infer or calculate any values.
- Every metric must include the exact source snippet from the document.
- If a value is ambiguous, set confidence below 0.8.
- Include page_number if determinable.

Respond ONLY with a JSON array. No preamble, no markdown fences.

Item schema:
{{
  "metric_name": "<name>",
  "metric_value": <number or null>,
  "metric_text": "<raw text representation>",
  "unit": "<EUR_M | USD_M | % | bps | x | null>",
  "segment": "<segment or null>",
  "geography": "<region or null>",
  "source_snippet": "<verbatim extract>",
  "page_number": <int or null>,
  "confidence": <0.0-1.0>
}}

--- DOCUMENT TEXT ---
{text}
"""

# ─────────────────────────────────────────────────────────────────
# Guidance Extractor
# ─────────────────────────────────────────────────────────────────
GUIDANCE_EXTRACTOR = """\
You are a guidance extraction agent. Identify forward-looking management
guidance statements from the text below.

Respond ONLY with a JSON array. No preamble, no markdown fences.

Item schema:
{{
  "metric_name": "<metric being guided>",
  "guidance_type": "range" | "point" | "directional",
  "guidance_text": "<full guidance statement>",
  "low": <number or null>,
  "high": <number or null>,
  "unit": "<unit or null>",
  "source_snippet": "<verbatim extract>",
  "confidence": <0.0-1.0>
}}

--- DOCUMENT TEXT ---
{text}
"""

# ─────────────────────────────────────────────────────────────────
# Thesis Comparator
# ─────────────────────────────────────────────────────────────────
THESIS_COMPARATOR = """\
You are a thesis comparison agent. Compare the new quarterly data
against the existing investment thesis.

CURRENT THESIS:
{thesis}

NEW QUARTER DATA:
{quarter_data}

PRIOR QUARTER DATA:
{prior_data}

Respond ONLY with a JSON object. No preamble, no markdown fences.

Schema:
{{
  "thesis_direction": "strengthened" | "weakened" | "unchanged",
  "supporting_signals": ["..."],
  "weakening_signals": ["..."],
  "new_risks": ["..."],
  "unresolved_questions": ["..."],
  "summary": "<2-3 sentence summary>"
}}
"""

# ─────────────────────────────────────────────────────────────────
# Surprise Detector
# ─────────────────────────────────────────────────────────────────
SURPRISE_DETECTOR = """\
You are a surprise detection agent. Identify results that deviate
meaningfully from prior expectations or consensus.

PRIOR EXPECTATIONS / GUIDANCE:
{expectations}

ACTUAL RESULTS:
{actuals}

Respond ONLY with a JSON array. No preamble, no markdown fences.

Item schema:
{{
  "metric_or_topic": "<what was surprising>",
  "direction": "positive" | "negative",
  "magnitude": "minor" | "major",
  "description": "<explanation>",
  "source_snippet": "<evidence>"
}}
"""

# ─────────────────────────────────────────────────────────────────
# IR Question Generator
# ─────────────────────────────────────────────────────────────────
IR_QUESTION_GENERATOR = """\
You are an IR question generation agent. Produce sharp, specific
follow-up questions an analyst should ask management on the next IR call.

COMPANY: {company}
PERIOD: {period}
KEY FINDINGS:
{findings}

THESIS:
{thesis}

Respond ONLY with a JSON array. No preamble, no markdown fences.

Item schema:
{{
  "topic": "<broad topic>",
  "question": "<the actual question>",
  "rationale": "<why this matters for the thesis>"
}}

Generate 5-8 questions.
"""

# ─────────────────────────────────────────────────────────────────
# One-Page Briefing
# ─────────────────────────────────────────────────────────────────
ONE_PAGE_BRIEFING = """\
You are a research briefing agent. Produce a concise one-page internal
research update using the structure below.

COMPANY: {company} ({ticker})
PERIOD: {period}

EXTRACTED KPIs:
{kpis}

THESIS COMPARISON:
{thesis_comparison}

SURPRISES:
{surprises}

Respond ONLY with a JSON object. No preamble, no markdown fences.

Schema:
{{
  "what_happened": "<summary of the quarter>",
  "what_changed": "<key changes vs prior>",
  "thesis_status": "<impact on thesis>",
  "risks": "<updated risk picture>",
  "follow_ups": "<open items>",
  "bottom_line": "<1-2 sentence conclusion>"
}}
"""
