# Robert — AI Medical Assistant Backend

FastAPI backend for the Robert AI Medical Assistant (Iteration 4). Provides GDPR-compliant chat over patient EMR data, powered by a SNOMED-grounded RAG pipeline.

## Architecture overview

```
User query
  → TermExtractor (Qwen2.5-1.5B, two-pass LLM)
      → categories (broad intent)  →  direct EMR section filter
      → terms      (specific intent) →  SapBERT CUI search
                                         → SNOMED graph expansion
                                           → EMR section matching
  → Prompt assembly → Gemini LLM → Response
```

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.12+ |
| pip | latest |
| GPU (optional) | CUDA-capable, ≥ 4 GB VRAM recommended |

**Data files** — place both in the project root (one level above `backend/`):

- `GraphModel_SNOMED_CUI_Embedding.pkl` — SapBERT CUI embeddings
- `SNOMED_CUI_MAJID_Graph_wSelf.pkl` — SNOMED knowledge graph

The server will not start without these files.

## Environment setup

Copy the example env file and fill in your API keys:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Google Gemini API key for chat responses and context compaction |
| `SARVAM_API_KEY` | No | Sarvam AI key for translation and TTS endpoints |

## Installation

```bash
cd backend

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt

# spaCy model required by Presidio for PII detection
python -m spacy download en_core_web_lg
```

## Running the backend

```bash
# From the backend/ directory with .venv active
uvicorn app.main:app --host 0.0.0.0 --port 8013 --reload
```

The API will be available at `http://localhost:8013`.  
Interactive docs: `http://localhost:8013/docs`

**Startup loads** (takes 30–90 s on first run):
- SapBERT embedding index
- SNOMED knowledge graph
- Qwen2.5-1.5B term extractor
- Presidio PII anonymizer

## Running the CLIs

All CLIs are run from the `backend/` directory with the virtual environment active.

### CUI semantic search

Map natural language to UMLS concepts via SapBERT embeddings.

```bash
# Single query
python -m app.cli.cui_search "my head hurts"

# Show extracted intent, categories, and terms
python -m app.cli.cui_search "diabetes medications" --show-terms

# Term extraction only (no CUI search, no embedding index needed)
python -m app.cli.cui_search "show my full medication history" --terms-only

# Interactive REPL
python -m app.cli.cui_search --interactive

# Custom result count and similarity threshold
python -m app.cli.cui_search "chest pain" --top-k 5 --threshold 0.8

# Custom embedding file
python -m app.cli.cui_search "diabetes" --embeddings /path/to/embeddings.pkl
```

### Graph expansion

Phase 1 (CUI extraction) → Phase 2 (SNOMED BFS expansion).

```bash
# Full pipeline from natural language query
python -m app.cli.graph_expand "my head hurts"

# Custom BFS depth and neighbour cap
python -m app.cli.graph_expand "chest pain" --depth 1 --max-per-hop 20

# Skip Phase 1 — pass CUI codes directly
python -m app.cli.graph_expand --cuis C0018681 C0008031

# Use all 108 SNOMED relations (no diagnostic filter)
python -m app.cli.graph_expand "headache" --all-relations
```

### Full RAG pipeline (EMR retrieval)

Runs all four phases against a patient EMR file and prints matched sections.

```bash
# Basic usage
python -m app.cli.emr_retrieve "my head hurts" --emr data/patient101.json

# Show the assembled LLM prompt
python -m app.cli.emr_retrieve "diabetes and blood sugar" --emr data/patient101.json --show-prompt

# Tune retrieval parameters
python -m app.cli.emr_retrieve "kidney function" \
  --emr data/patient101.json \
  --top-k 5 \
  --threshold 0.75 \
  --depth 1 \
  --match-threshold 0.4
```

## Running tests

```bash
# All tests (excludes endpoint integration tests that require a running server)
python -m pytest tests/ --ignore=tests/test_endpoints.py -v

# Endpoint integration tests (requires server running on port 8013)
python -m pytest tests/test_endpoints.py -v

# Specific test file
python -m pytest tests/test_pipeline.py -v
```

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/chat` | Send a message, get a response |
| `GET` | `/sessions/{id}` | Retrieve session history |
| `DELETE` | `/sessions/{id}` | Delete session (GDPR Art. 17) |
| `GET` | `/gdpr/export/{id}` | Export personal data (GDPR Art. 15) |
| `POST` | `/gdpr/rectify/{id}` | Rectify personal data (GDPR Art. 16) |

Full schema available at `/docs` when the server is running.

## Project structure

```
backend/
├── app/
│   ├── cli/
│   │   ├── cui_search.py       # CUI semantic search CLI
│   │   ├── graph_expand.py     # Graph expansion CLI
│   │   └── emr_retrieve.py     # Full pipeline CLI
│   ├── models/
│   │   └── chat_models.py      # Pydantic request/response models
│   ├── routers/
│   │   ├── chat.py             # POST /chat
│   │   ├── sessions.py         # Session management
│   │   └── gdpr_router.py      # GDPR endpoints
│   ├── services/
│   │   ├── rag/
│   │   │   ├── term_extractor.py   # Two-pass LLM extraction (Qwen2.5)
│   │   │   ├── cui_search.py       # SapBERT FAISS search
│   │   │   ├── embeddings.py       # EmbeddingIndex loader
│   │   │   ├── graph.py            # KnowledgeGraph loader
│   │   │   ├── graph_expand.py     # SNOMED BFS expansion
│   │   │   ├── emr.py              # EMR JSON parser
│   │   │   ├── emr_match.py        # CUI-to-section matching
│   │   │   ├── prompt.py           # Prompt assembly
│   │   │   └── pipeline.py         # 4-phase RAG orchestrator
│   │   ├── gemini_service.py       # Gemini LLM client
│   │   ├── presidio_anonymizer.py  # PII de-identification
│   │   ├── context_compaction.py   # History summarisation
│   │   └── session_store.py        # Session persistence
│   └── main.py                 # FastAPI app + startup
├── tests/
├── scripts/
│   └── benchmark.py            # RAG pipeline benchmarking
├── .env.example
├── requirements.txt
└── README.md
```
