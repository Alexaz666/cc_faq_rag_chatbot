# Segment-Aware Credit Card FAQ RAG Bot

A retrieval-augmented chatbot built with FastAPI, Chroma, and OpenAI models, featuring metadata-aware segment filtering, multi-stage guardrails, and a structured evaluation harness.

This project explores how to design a domain-specific FAQ assistant with production-style safety controls and evaluation-driven iteration.

---

## Motivation

This project was built to explore guardrail-first RAG design in a structured financial product domain. The goal was not just to retrieve and generate answers, but to:

- Control model behaviour across different product segments (personal vs business).
- Enforce refusal when appropriate.
- Evaluate system behaviour systematically rather than relying on ad-hoc testing.

All knowledge is derived from publicly available product pages. No internal systems or proprietary data were used.

---

## What This Demonstrates

- **Retrieval-Augmented Generation (RAG)** using Chroma vector store.
- **Segment-aware filtering** (`personal`, `business`, `generic`) at retrieval time.
- **Query normalization & preprocessing** via `prep_retrieval`.
- **Multi-stage guardrails**, including:
  - Empty retrieval detection
  - Relevance threshold gating
  - Post-generation refusal validation
- **Evaluation harness with diagnostics**, including:
  - Bucketed test cases
  - Guardrail behaviour validation
  - Segment integrity checks
  - Retrieval preview logging
  - Manual answer validation workflow

---

## Architecture

### Ingestion Pipeline

```text
Public Web Pages
        ↓
Scrape → Clean → Chunk
        ↓
Embed (text-embedding-3-small)
        ↓
Persist in Chroma
```

### Query Pipeline

```text
User Query + Segment
        ↓
prep_retrieval()
        ↓
Similarity Search (with metadata filter)
        ↓
Relevance Gate (threshold check)
        ↓
LLM (gpt-4-turbo)
        ↓
Post-answer Guardrail (refusal check)
        ↓
Response
```

---

## Key Design Decisions

### 1. Segment-Aware Retrieval

Each chunk is tagged with metadata (`personal`, `business`, `generic`).  
At query time, retrieval is filtered to only return:

- The requested segment
- Generic content

This prevents cross-contamination between personal and business products.

---

### 2. Guardrail-First Design

Before calling the LLM:

- If no documents are retrieved → refuse.
- If top similarity score is below a threshold → refuse.

After LLM generation:

- If the answer matches the refusal template → classify as refusal.

This ensures behaviour is deterministic and measurable.

---

### 3. Query Normalization

The `prep_retrieval` module:

- Normalizes segment inputs.
- Adjusts retrieval query formulation.
- Builds metadata filters.
- Tracks query notes for diagnostics.

This enables consistent retrieval behaviour and better debugging.

---

## Evaluation Design

The evaluation set contains **30 questions** across four buckets:

| Bucket | Count | Expected Behaviour |
|--------|-------|-------------------|
| Answerable | 18 | Provide answer |
| Unanswerable | 5 | Refuse |
| Out-of-scope | 5 | Refuse |
| Segment-conflict | 2 | Refuse |

### Breakdown of Answerable Questions

- 5 general credit card questions
- 8 personal/retail product questions
- 5 business product questions

---

## Evaluation Results

- **100% guardrail behaviour pass rate**
  - All questions were correctly answered or refused as expected.
- **Segment integrity validated**
  - Retrieved documents matched requested segment.
- **Manual review of answer quality**
  - 4 of 18 answerable responses were flagged for factual imprecision or undesirable phrasing.

Evaluation artifacts included in this repo:

- `eval_set2.json`
- `eval_set2_results.json`
- `eval_set2_review.csv`
- `eval_set2_guardrail_fail.json`
- `eval_set2_segment_mismatch.json`

---

## Models Used

- **Embedding Model:** `text-embedding-3-small`  
- **LLM:** `gpt-4-turbo`

---

## How to Run

> Run all commands from project root with virtual environment activated.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:

```
OPENAI_API_KEY=your_key_here
```

### 3. Build Index

```bash
python -m src.build_index
```

### 4. Start API

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Navigate to:

```
http://127.0.0.1:8000/docs
```

Use the `/ask` endpoint to test queries.

### 5. Run Evaluation

```bash
python -m eval.run_eval
```

---

## Planned Enhancements

- Frontend UI with explicit segment selection
- Segment conflict clarification logic
- Product name injection into chunks
- Secondary retrieval pass within top sources

---

## Limitations

- Knowledge base derived from scraped public webpages.
- Public webpage structure introduces formatting inconsistencies affecting chunk quality.
- Not deployed to production.
- No authentication or rate limiting implemented.

---

## Disclaimer

This is an educational portfolio project built using publicly available product pages.  
No internal systems, proprietary data, or confidential information were used.

This project is not affiliated with or endorsed by Commonwealth Bank.