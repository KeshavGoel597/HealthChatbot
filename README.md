# Robert AI: Medical Assistant (DASS Spring 2026)

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/4T_GxXnv)

[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=22387623&assignment_repo_type=AssignmentRepo)

---

# Robert AI

Robert AI is an advanced AI-powered medical assistant built around a **SNOMED-grounded Retrieval-Augmented Generation (RAG) pipeline**.

The system leverages Electronic Medical Records (EMRs), medical knowledge graphs, and Large Language Models (LLMs) to provide context-aware and medically relevant responses while maintaining strong privacy and compliance guarantees.

The project consists of:

* **FastAPI Backend** for orchestration, retrieval, GDPR operations, and model integration.
* **Next.js Frontend** for multilingual chat, voice interaction, and evidence visualization.
* **SNOMED Knowledge Graph + SapBERT Retrieval Layer** for clinically grounded reasoning.

---

# 🌟 Key Features

## SNOMED-Grounded RAG Pipeline

* SapBERT-based concept retrieval
* FAISS-powered CUI search
* SNOMED knowledge graph expansion
* EMR evidence retrieval and ranking
* Context-aware prompt construction

## Multi-Model LLM Support

Supports multiple inference backends:

* Gemini
* Qwen
* MedGemma (via Ollama)

Allowing flexible deployment depending on latency, privacy, and cost requirements.

## Voice & Multilingual Capabilities

Integrated with Sarvam AI for:

* Speech-to-Text (STT)
* Text-to-Speech (TTS)
* Translation
* Multilingual interaction

## GDPR Compliance & Privacy

Built-in support for:

* Article 15 (Access & Transparency)
* Article 17 (Right to be Forgotten)
* Session deletion
* Evidence timelines
* Patient data protection

## Evidence Transparency

The frontend explicitly displays:

* Retrieved evidence
* Supporting EMR fields
* Reasoning context

allowing users to understand how responses were generated.

---

# 🧠 RAG Architecture

```text
User Query
      │
      ▼
SapBERT Encoding
      │
      ▼
CUI Retrieval (FAISS)
      │
      ▼
SNOMED Graph Expansion
      │
      ▼
EMR Evidence Retrieval
      │
      ▼
Prompt Assembly
      │
      ▼
Gemini / Qwen / MedGemma
      │
      ▼
Response + Evidence Panel
```

The retrieval layer uses SNOMED-grounded semantic search to identify clinically relevant concepts and evidence before generating a response.

---

# 🏗 Repository Layout

```text
.
├── .github/
│   └── workflows/
│       ├── snapshot-integrity.yml
│       └── ...
│
├── docs/
│   ├── StatusTracker.xls
│   ├── ProjectPlan/
│   ├── MinutesOfMeetings/
│   └── ...
│
├── src/
│   ├── backend/
│   │   ├── app/
│   │   ├── data/
│   │   └── ...
│   │
│   ├── frontend/
│   │   ├── app/
│   │   ├── components/
│   │   └── ...
│   │
│   ├── GraphModel_SNOMED_CUI_Embedding.pkl
│   ├── SNOMED_CUI_MAJID_Graph_wSelf.pkl
│   └── sm_t047_cui_aui_eng.pkl
│
├── flake.nix
├── .envrc
└── README.md
```

---

# 📂 Required Data Files

To run the RAG pipeline locally, the following files must be placed directly inside the `src/` directory.

| File                                  | Required | Purpose                     |
| ------------------------------------- | -------- | --------------------------- |
| `GraphModel_SNOMED_CUI_Embedding.pkl` | Yes      | SapBERT CUI embedding index |
| `SNOMED_CUI_MAJID_Graph_wSelf.pkl`    | Yes      | SNOMED knowledge graph      |
| `sm_t047_cui_aui_eng.pkl`             | Optional | CUI → concept name mapping  |

Directory structure:

```text
src/
├── GraphModel_SNOMED_CUI_Embedding.pkl
├── SNOMED_CUI_MAJID_Graph_wSelf.pkl
├── sm_t047_cui_aui_eng.pkl
├── backend/
└── frontend/
```

---

# 🚀 Getting Started

## 1. Backend Setup (FastAPI)

### Prerequisites

* Python 3.12+
* uv (recommended)
* Ollama (optional)

### Configure Environment

```bash
cd src/backend

cp .env.example .env
```

Add required API keys:

```env
GEMINI_API_KEY=<your-key>
SARVAM_API_KEY=<your-key>
```

### Install Dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync --extra dev

uv run python -m spacy download en_core_web_lg
```

### Optional: Local MedGemma via Ollama

```bash
ollama pull MedAIBase/MedGemma1.5:4b
```

### Run Backend

```bash
uv run uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8013 \
    --reload
```

Backend URL:

```text
http://localhost:8013
```

### macOS Note

If you encounter OpenMP runtime conflicts:

```bash
KMP_DUPLICATE_LIB_OK=TRUE \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1
```

---

## 2. Frontend Setup (Next.js)

### Prerequisites

* Node.js 20+
* npm or bun

### Install Dependencies

```bash
cd src/frontend

npm install
```

### Run Development Server

```bash
npm run dev
```

Frontend URL:

```text
http://localhost:3000/chat
```

---

# 🛠 CLI Tools

The backend includes utilities for testing and debugging the retrieval pipeline.

Navigate to:

```bash
cd src/backend
```

## CUI Search

```bash
uv run python -m app.cli.cui_search \
    "my head hurts" \
    --show-terms
```

## Graph Expansion

```bash
uv run python -m app.cli.graph_expand \
    "chest pain" \
    --depth 1 \
    --max-per-hop 20
```

## EMR Retrieval

```bash
uv run python -m app.cli.emr_retrieve \
    "my head hurts" \
    --emr data/patient101.json \
    --show-prompt
```

---

# 🧪 Testing

Run backend tests:

```bash
uv run pytest -v
```

---

# 🔒 GDPR & Privacy

Robert AI is designed with privacy and compliance in mind.

## Supported Features

* GDPR Article 15 — Access & Transparency
* GDPR Article 17 — Right to be Forgotten
* Session deletion
* Evidence timelines
* Session auditability

## PII Protection

Patient-identifiable information should be anonymized before external model access.

Supported mechanisms include:

* Microsoft Presidio
* Local PII detection
* Evidence filtering
* Secure session handling

No patient-identifiable information should be transmitted to external services unless explicitly approved by deployment policies.

---

# 💡 Roadmap

## 1. Dynamic Patient Authentication

### Current State

* Frontend currently uses a hardcoded patient identifier.

### Future Work

* Login system
* Patient selection portal
* React Context or Zustand-based state management

---

## 2. Frontend Proxy Routing

### Current State

Frontend components may directly reference backend URLs.

### Future Work

* Use Next.js rewrite rules
* Remove CORS dependencies
* Environment-specific backend routing

---

## 3. Containerization

### Current State

Services must be started manually.

### Future Work

* Dockerfiles
* Docker Compose
* One-command deployment

---

## 4. Database Migration

### Current State

EMRs and sessions are stored using local files.

### Future Work

* PostgreSQL
* SQLAlchemy / SQLModel
* Vector databases such as:

  * Qdrant
  * Milvus
  * Pinecone

---

## 5. End-to-End Testing

### Current State

Backend testing exists through Pytest.

### Future Work

* Playwright
* Cypress
* Full user workflow testing

---

# 📚 Academic & Course Context

This repository is built on top of the DASS Spring 2026 project template.

## Status Tracker

Weekly updates must be recorded in:

```text
docs/StatusTracker.xls
```

Do not convert the tracker to CSV format.

---

## Automated Snapshots

Every Friday, GitHub Actions automatically:

* Creates an immutable tag (`submission-week-N`)
* Generates a release
* Stores hash manifests for `src/` and `docs/`

---

## Process Integrity

Automated integrity checks compare:

* Weekly tracker entries
* Git commit history

Missing supporting commits may trigger an audit review.

The integrity workflow is implemented in:

```text
.github/workflows/snapshot-integrity.yml
```

---

# 👥 Authors

Developed as part of the DASS Spring 2026 course project.

Student Team:

* Keshav
* Vikesh
* Advait
* Amish
* Venya

---
