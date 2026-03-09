# CUI Semantic Search — Implementation Document

## Overview

This module extracts **UMLS Concept Unique Identifiers (CUIs)** from natural language queries using semantic embedding similarity. Unlike string-matching tools like QuickUMLS, this approach handles colloquial and informal medical language (e.g. "my head hurts" → `C0018681` Headache).

## Problem

Standard NLP-to-CUI pipelines (QuickUMLS, MetaMap, cTAKES) rely on lexical matching — they compare surface-level tokens against a dictionary of concept names. This fails on:

- Colloquial language: "my head hurts", "tummy ache", "can't breathe"
- Abbreviations and slang: "bp is high", "sugar levels off"
- Descriptions rather than terms: "I feel pressure in my chest"

We need **semantic** matching: understand what the user *means*, not just what words they used.

## Approach

We use the same embedding setup from the **DR.KNOWS** medical knowledge graph paper:

1. **SapBERT** (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`) — a biomedical language model fine-tuned specifically for UMLS concept linking. It maps both medical terms and free-text descriptions into the same 768-dimensional vector space.

2. **Pre-computed CUI embeddings** — DR.KNOWS provides `GraphModel_SNOMED_CUI_Embedding.pkl`, a dictionary mapping 407,288 SNOMED CUIs to their SapBERT embeddings (each a float32 vector of shape `(1, 768)`).

3. **FAISS IndexFlatIP** — all 407k CUI embeddings are L2-normalized and loaded into a FAISS inner-product index. Since vectors are unit-length, inner product equals cosine similarity. This gives exact nearest-neighbor search in ~70-100ms per query.

4. **CUI Vocabulary** — `CUI_Vocab.json` (actually pickle format) maps each CUI to its preferred text names from the UMLS Metathesaurus (489,354 entries). We pick the shortest name per CUI for display.

### Data Flow

```
User query ("my head hurts")
        │
        ▼
┌──────────────────┐
│  SapBERT Encoder │  tokenize + forward pass → pooler_output
│  (PubMedBERT)    │  → L2-normalize → 768-dim float32 vector
└──────────────────┘
        │
        ▼
┌──────────────────┐
│  FAISS Index     │  inner-product search over 407k CUI vectors
│  (IndexFlatIP)   │  → top-k (CUI, score) pairs
└──────────────────┘
        │
        ▼
┌──────────────────┐
│  Threshold Filter │  keep only results with score ≥ threshold (default 0.7)
└──────────────────┘
        │
        ▼
┌──────────────────┐
│  Name Resolution │  CUI_Vocab.json lookup → human-readable names
└──────────────────┘
        │
        ▼
Output: [{"cui": "C0018681", "name": "Headache", "score": 0.8565}, ...]
```

## File Structure

```
backend/app/
├── services/rag/
│   ├── __init__.py          # Package marker (empty)
│   ├── embeddings.py        # Core: EmbeddingIndex class (~120 lines)
│   └── cui_search.py        # Search function + pretty printer (~50 lines)
├── cli/
│   ├── __init__.py          # CLI logic: argparse, interactive mode (~100 lines)
│   └── cui_search.py        # Entry point: python -m app.cli.cui_search
└── main.py                  # FastAPI app (loads EmbeddingIndex at startup)
```

### Data Files (project root, not committed)

| File | Format | Contents |
|------|--------|----------|
| `GraphModel_SNOMED_CUI_Embedding.pkl` | pickle | Dict[str, ndarray(1,768)] — 407,288 CUI embeddings |
| `SNOMED_CUI_MAJID_Graph_wSelf.pkl` | pickle | networkx DiGraph — 407,288 nodes, 3.4M edges (for future graph expansion) |
| `CUI_Vocab.json` | pickle (despite name) | Dict[str, list[[aui_id, text]]] — 489,354 CUI name entries |

## Module Details

### `embeddings.py` — EmbeddingIndex

The core class. Load once, query many times.

**`__init__(embedding_path, vocab_path=None)`**
1. Loads `GraphModel_SNOMED_CUI_Embedding.pkl` — unpickles the dict, stacks all CUI vectors into a `(407288, 768)` float32 matrix
2. L2-normalizes every row (so inner product = cosine similarity)
3. Creates `faiss.IndexFlatIP(768)` and adds all rows
4. Loads `CUI_Vocab.json` (auto-detects if in same directory as embeddings) — extracts the shortest preferred name per CUI
5. Loads SapBERT tokenizer + model from HuggingFace (cached after first download)

**`encode(text) → ndarray(768,)`**
- Tokenizes input with SapBERT tokenizer (max 256 tokens, truncation)
- Forward pass through BertModel → takes `pooler_output` (the [CLS] representation)
- L2-normalizes the resulting vector

**`search(query_vec, top_k=10) → list[(cui, score)]`**
- Reshapes query to `(1, 768)`, calls `faiss.IndexFlatIP.search()`
- Returns list of `(CUI string, cosine similarity score)` tuples in descending order

**`get_name(cui) → str`**
- Looks up the CUI in the vocabulary dict. Returns the shortest name, or the CUI itself if not found.

### `cui_search.py` — Search API

**`find_cuis(query, index, top_k=10, threshold=0.7) → list[dict]`**
- Encodes query → searches index → filters by threshold
- Returns `[{"cui": str, "name": str, "score": float}]`

**`print_results(query, results)`**
- Formatted table output for CLI/debugging

### `cli/__init__.py` — CLI Module

The entry point is `python -m app.cli.cui_search` (from the `backend/` directory).

**Modes:**
- **Single query:** `python -m app.cli.cui_search "my head hurts"`
- **Interactive REPL:** `python -m app.cli.cui_search --interactive`

**Flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `-k` / `--top-k` | 10 | Number of results to return |
| `-t` / `--threshold` | 0.7 | Minimum cosine similarity score |
| `--embeddings` | auto-detected | Path to CUI embedding pickle |
| `--vocab` | auto-detected | Path to CUI_Vocab.json |

### `main.py` — FastAPI Integration

The `EmbeddingIndex` is loaded once during app startup via the `lifespan` context manager and stored in `app.state.embedding_index`. This avoids reloading the ~1.2GB of data on every request.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.embedding_index = EmbeddingIndex(_EMBEDDING_PATH)
    yield
```

Any route handler can access it via `request.app.state.embedding_index`.

## Performance

| Metric | Value |
|--------|-------|
| Index load time | ~6 seconds (model + 407k embeddings + vocab) |
| Model size (SapBERT) | ~440MB (cached on disk after first download) |
| Embedding file size | ~1.2GB (407k × 768 × float32) |
| Query latency | 70-100ms per query (encode + FAISS search) |
| FAISS index type | `IndexFlatIP` — exact search, no approximation |

## Example Results

```
Query: "my head hurts"
  Rank  CUI          Score    Name
  ───── ──────────── ──────── ────────────────────────────────────────
  1     C0018681     0.8565   Headache
  2     C1534966     0.8171   ([D]Pain: (in head NOS) or (jaw))
  3     C0578055     0.8127   Pain of head and neck region

Query: "chest pain"
  → C0008031 (Chest Pain), C0232288 (Chest wall pain), ...

Query: "can't breathe"
  → C0013404 (Dyspnea), C0231800 (Breathlessness), ...
```

## Dependencies

- `torch` — PyTorch (CPU-only is sufficient)
- `transformers` — HuggingFace (for SapBERT model loading)
- `faiss-cpu` — Facebook AI Similarity Search
- `numpy` — array operations

## Why SapBERT Instead of QuickUMLS?

| | QuickUMLS | SapBERT Semantic Search |
|---|---|---|
| Matching | Lexical (string similarity) | Semantic (meaning similarity) |
| "my head hurts" → Headache | ✗ No match | ✓ Score: 0.856 |
| "can't breathe" → Dyspnea | ✗ No match | ✓ Score: ~0.84 |
| "blood sugar" → Glucose | ✗ Partial/wrong | ✓ Score: ~0.80 |
| Speed | Fast (string index) | ~80ms (neural encode + FAISS) |
| Setup | Requires UMLS license download | Pre-computed embeddings from DR.KNOWS |

## Future Work (Not Yet Implemented)

- **Graph expansion (Phase 2):** BFS over the SNOMED knowledge graph (`SNOMED_CUI_MAJID_Graph_wSelf.pkl`) to expand retrieved CUIs to related concepts
- **EMR retrieval (Phase 3):** Use expanded CUI sets to retrieve relevant sections from patient EMR data
- **Prompt assembly (Phase 4):** Combine retrieved context into structured prompts for the LLM
