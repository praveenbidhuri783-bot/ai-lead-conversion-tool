# AI Lead Sentiment & Conversion Scoring Tool

A production-ready Streamlit application that analyses AI voice-call summaries, scores every
lead's conversion likelihood, and recommends the next-best agent action and pitch.

## Features

- Upload `.xlsx`, `.xls`, or `.csv` files with automatic call-summary column detection
- Configurable, transparent rule-based scoring engine (0-100), no LLM required
- Hard exclusion rules (wrong number, DNC, not interested, etc.)
- Lead prioritisation into P1 Hot / P2 Warm / P3 Nurture / P4 Exclude
- Sentiment classification, objection detection, and objection-specific agent pitches
- Executive dashboard with KPIs, filters, and 10 charts including a conversion funnel
- Multi-sheet Excel export (Executive Summary, All Leads, P1-P4, Objection Analysis, Scoring Methodology)
- Optional LLM enrichment (Anthropic or OpenAI) for top-scoring leads, fully optional
- Mobile-number masking in the dashboard (`XXXXXX1234`)

## Project Structure

```text
ai-lead-conversion-tool/
├── app.py                     # Streamlit entry point
├── requirements.txt
├── Dockerfile
├── .env.example
├── config/
│   ├── scoring_config.json    # Base score, positive/negative signals, hard exclusions, thresholds
│   ├── pitch_config.json      # Agent pitches by priority/objection
│   └── column_mapping.json    # Input column aliases
├── modules/
│   ├── file_reader.py
│   ├── validator.py
│   ├── scorer.py
│   ├── sentiment.py
│   ├── objection_detector.py
│   ├── pitch_generator.py
│   ├── dashboard.py
│   ├── exporter.py
│   └── llm_classifier.py      # Optional AI enrichment
├── assets/
│   └── sample_input.xlsx
├── scripts/
│   └── generate_sample.py
└── tests/
    ├── test_scorer.py
    ├── test_validator.py
    ├── test_objection_detector.py
    └── test_exporter.py
```

## Local Setup

```bash
cd ai-lead-conversion-tool
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`). Try it immediately with
`assets/sample_input.xlsx`.

## Running Tests

```bash
pytest tests/ -v
```

## Optional: Advanced AI Analysis

The tool works fully offline with rule-based scoring. To enable LLM enrichment for top leads:

```bash
cp .env.example .env
# then set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env, and export it, e.g.:
export ANTHROPIC_API_KEY=sk-...
streamlit run app.py
```

Toggle **"Use Advanced AI Analysis"** in the Configuration section. Only masked call-summary
text (phone numbers redacted) is sent to the LLM, and only for the top-N leads you choose —
rule-based scoring remains authoritative for every record.

## Configuring Scoring Without Code Changes

Business users can edit these files directly (no redeploy required if mounted as a volume):

- `config/scoring_config.json` — base score, keyword signals, hard exclusions, priority/sentiment
  thresholds, SLA rules, objection keywords
- `config/pitch_config.json` — agent pitch text by priority, objection, service, and language
- `config/column_mapping.json` — column-name aliases the file reader recognises automatically

Priority thresholds and rule groups can also be adjusted live from the app's Configuration
section for a single run, without touching the JSON files.

## Docker

```bash
docker build -t ai-lead-conversion-tool .
docker run -p 8501:8501 --env-file .env ai-lead-conversion-tool
```

## Production Deployment Notes

- **Streamlit Community Cloud**: point it at this repo/`app.py`; add API keys as secrets if using
  Advanced AI Analysis.
- **Azure App Service / Container Apps, AWS ECS**: build and push the Docker image above; set
  `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` as environment secrets if needed; expose port `8501`.
- No external AI API is required for core functionality — only set keys if you want LLM
  enrichment.
- The app avoids logging full phone numbers; only masked numbers are shown in the dashboard.
  Full numbers appear in downloaded Excel/CSV files and in the dashboard only when the
  "Reveal full mobile numbers" checkbox is explicitly enabled.

## Scoring Logic Summary

Every lead starts at a base score of 25. Configured positive keyword/column signals add points
(e.g. "wants to book" +25, callback requested +22); negative signals subtract points (e.g.
"wrong number" -60, "not interested" -45). The final score is clamped to 0-100. Hard exclusions
(wrong number, DNC, not interested, not eligible, fraud, duplicate completed transaction,
already-availed-no-rebooking) always force the lead to **P4 - Exclude**, regardless of score.

| Priority | Score Range | SLA |
|---|---|---|
| P1 - Hot | ≥ 75 | Agent call within 30 minutes |
| P2 - Warm | 55–74 | Agent follow-up within 4 hours |
| P3 - Nurture | 35–54 | WhatsApp/SMS, retry within 24-48 hours |
| P4 - Exclude | < 35 or hard-excluded | Not prioritised |

See `config/scoring_config.json` for the exact, editable rule set, and the "Scoring Methodology"
sheet in every exported report for the rules actually used in that run.
