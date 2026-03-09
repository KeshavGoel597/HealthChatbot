# Robert — AI Medical Assistant (Iteration 2)

A GDPR-compliant, context-compaction-aware healthcare chatbot that helps patients understand their Electronic Medical Records (EMR) using AI. Robert is an AI assistant — **not a licensed physician** — and all responses must be verified with a qualified healthcare professional.

---

## ⚡ Quick Start — Running Iteration 2

### Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | ≥ 3.11 | For the backend |
| Node.js | ≥ 18 (or Bun) | For the frontend |
| GEMINI_API_KEY | — | Required if using Gemini model |
| HuggingFace token | — | Required only for local HF / MedGemma models |

---

### 1. Backend Setup

```bash
# Navigate to the backend directory
cd src/iteration2/backend

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn python-dotenv google-genai transformers torch accelerate sentencepiece
```

Create a `.env` file in `src/iteration2/backend/`:
```env
# Required for Gemini model (default)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: only needed if you use MedGemma or Qwen local models
# Log in via: huggingface-cli login
```

Start the backend:
```bash
# From src/iteration2/backend/
uvicorn app.main:app --host 0.0.0.0 --port 8013 --reload
```

The backend will be available at `http://localhost:8013`.  
Interactive API docs: `http://localhost:8013/docs`

> **Note:** On every startup, the backend automatically runs a **GDPR retention cleanup** — any chat sessions older than 30 days are permanently deleted.

---

### 2. Frontend Setup

```bash
# Navigate to the frontend directory
cd src/iteration2/frontend

# Install dependencies (using Bun — recommended)
bun install
# OR using npm:
# npm install

# Start the development server
bun dev
# OR: npm run dev
```

The frontend will be available at `http://localhost:3000`.

---

### 3. Patient Data

EMR data files are stored as JSON in `src/iteration2/backend/data/`.  
A sample file `patient101.json` is included. The system reads this **read-only** — it never writes to EMR files.

To add a new patient, place a JSON file named `{patient_id}.json` in the `data/` directory.

---

## 🤖 Model Selection

Robert supports three LLM backends, selectable from the UI:

| Model | Type | Notes |
|---|---|---|
| **Gemini 2.5 Flash** | Cloud API | Default. Requires `GEMINI_API_KEY`. Uses Gemini-based context compaction. |
| **MedGemma 4B** | Local (GPU) | Gated model — requires HuggingFace login. ~4GB VRAM. |
| **Qwen 0.5B** | Local (CPU/GPU) | Small model, low resource usage. Uses deterministic compaction. |

> **For production**: The client intends to self-host a local LLM (MedGemma or equivalent). This ensures patient data is processed entirely within the healthcare infrastructure and is never transmitted to third-party AI providers.

---

## 🔐 GDPR Compliance Features

This implementation addresses the following GDPR requirements:

### Consent & Transparency (Art. 5(1)(a), Art. 7)
- A **GDPR consent modal** appears before any chat begins
- Patients explicitly opt in (or out) of:
  - EMR data access during the session
  - Storing the conversation for future reference
- A persistent **AI disclaimer banner** is shown throughout the interface
- Every response ends with a reminder that Robert is an AI, not a physician

### Data Minimisation (Art. 5(1)(c))
- **Context Compaction**: Only the most relevant recent turns are sent to the LLM — older turns are summarised into a compact clinical note. See [Context Compaction](#-context-compaction) below.
- The clinical summary injected into prompts is extracted minimally (only fields relevant to the query)

### Storage Limitation (Art. 5(1)(e))
- Chat history is **only saved if the patient explicitly opts in** (default: not stored)
- All sessions have an **auto-deletion date** (30 days from creation)
- Retention cleanup runs **automatically on every backend startup**
- Stored chats contain only conversation text — **no raw EMR data** is ever saved in chat logs

### Right of Access (Art. 15)
- Each assistant message has a toggleable **📋 EMR Evidence** panel
- Patients can see exactly which EMR data categories (e.g., "Medical Diagnoses", "Lab Results") influenced each AI response

### Right to Rectification (Art. 16)
- Robert cannot modify EMR records — it is **read-only**
- If a patient notices incorrect information, Robert informs them to contact their healthcare provider or system administrator to request corrections

### Right to Erasure (Art. 17)
- Every session in the sidebar has a **🗑 Delete** button
- Deletes the session permanently via `DELETE /gdpr/sessions/{session_id}`
- Patients can also request deletion of all their sessions via `DELETE /gdpr/patient/{patient_id}` (API)

### Automated Decision-Making (Art. 22)
- Robert is explicitly framed as a **clinical decision-support tool**, not a diagnostic authority
- Every response includes a physician verification disclaimer
- The system prompt instructs the model to never claim to diagnose, prescribe, or replace medical judgment

---

## 🗜 Context Compaction

As conversations grow, sending the full history to the LLM on every turn becomes expensive and risks exceeding token limits. Context compaction addresses this:

**How it works:**
1. Before each LLM call, the total estimated token count of the history is checked
2. If it exceeds **3,000 tokens** (threshold), older turns are **compacted** into a clinical summary
3. Only the summary + the **last 6 messages** are sent to the LLM
4. The summary is saved in the session and grows incrementally

**Two compaction strategies:**
- **Gemini model**: Uses a second Gemini API call to produce a clinically faithful summary, preserving all symptoms, medications, dates, and concerns mentioned
- **Local models (Qwen/MedGemma)**: Uses deterministic regex-based extraction — no extra inference cost

**GDPR note**: The compacted summary only contains what the patient *said* during chat. It is not a copy of the EMR, satisfying Data Minimisation (Art. 5(1)(c)).

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/sessions/{patient_id}` | Create new session (accepts consent flags) |
| `GET` | `/sessions/{patient_id}` | List active sessions |
| `GET` | `/sessions/{session_id}/messages` | Get full session with messages |
| `POST` | `/sessions/{session_id}/message` | Send a message (with compaction + GDPR) |
| `POST` | `/gdpr/consent/{session_id}` | Update consent flags (GDPR Art. 5) |
| `GET` | `/gdpr/evidence/{session_id}` | Get EMR evidence log (GDPR Art. 15) |
| `DELETE` | `/gdpr/sessions/{session_id}` | Delete one session (GDPR Art. 17) |
| `DELETE` | `/gdpr/patient/{patient_id}` | Delete all sessions for patient (GDPR Art. 17) |
| `GET` | `/health` | Health check |

---

## 📁 Project Structure

```
iteration2/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app, GDPR startup cleanup
│   │   ├── models/
│   │   │   └── chat_models.py       # Pydantic models (with GDPR fields)
│   │   ├── routers/
│   │   │   ├── chat.py              # Simple /chat endpoint
│   │   │   ├── sessions.py          # Session management + compaction
│   │   │   └── gdpr_router.py       # GDPR endpoints (Art. 15, 17)
│   │   └── services/
│   │       ├── context_compaction.py # Compaction logic (shared)
│   │       ├── gemini_service.py    # Gemini LLM (cloud)
│   │       ├── huggingface_service.py # Qwen local LLM
│   │       ├── medgemma_service.py  # MedGemma local LLM
│   │       └── sarvam_service.py    # Translation + TTS
│   └── data/
│       ├── patient101.json          # Sample EMR data (read-only)
│       └── sessions/                # Stored chat sessions (created at runtime)
└── frontend/
    ├── app/                         # Next.js app router
    ├── components/
    │   └── chat/
    │       ├── consent-modal.tsx    # GDPR consent modal
    │       ├── chat-interface.tsx   # Main chat UI
    │       └── chat-sidebar.tsx     # Session list + delete
    └── ...
```

---

## ⚠ Important Notes

- **EMR data is never modified** by Robert — it is always read-only. Corrections must be made through the healthcare provider's system.
- **Local models require significant RAM/VRAM**: MedGemma (~8GB VRAM), Qwen-0.5B (~1GB RAM).
- **Sarvam API** is used for translation and TTS. Requires a `SARVAM_API_KEY` environment variable if used.
- In the final production version, a locally-hosted LLM will replace the Gemini cloud API, ensuring all patient data stays within the healthcare infrastructure.
