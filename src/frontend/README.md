# Robert Frontend (Next.js)

## Description

This frontend is the web UI for Robert, the AI Medical Assistant.  
It provides:

- chat UI with model selection (Gemini / MedGemma / Qwen)
- multilingual input/output support
- voice input (microphone + backend transcription)
- GDPR consent flow and session controls
- evidence panel showing EMR fields used in responses

## Files and directories

| Path | Purpose |
|---|---|
| `app/chat/page.tsx` | Main chat route (`/chat`) mounting the chat interface. |
| `components/chat/chat-interface.tsx` | Core chat logic, backend API calls, model/language selection, voice input, evidence UI. |
| `components/chat/chat-sidebar.tsx` | Session list, new chat, and delete chat actions. |
| `components/chat/consent-modal.tsx` | GDPR consent modal (EMR + history storage consent). |
| `components/theme-provider.tsx` | Theme provider wrapper (`next-themes`). |
| `components/mode-toggle.tsx` | Light/dark/system theme toggle. |
| `components/ui/` | Shadcn/Radix UI primitives used across the app. |
| `app/layout.tsx` | Global layout, metadata, fonts, theme provider wiring. |
| `app/globals.css` | Tailwind/shadcn styles and design tokens. |
| `next.config.ts` | Next config + backend proxy rewrite (`/api/python/:path*`). |
| `package.json` | Scripts and dependencies. |

## Prerequisites

- Node.js `20+`
- npm (recommended, `package-lock.json` is present)
- Backend running at `http://localhost:8013` (see `../backend/README.md`)

## Installation (npm)

Install npm packages:

```bash
cd src/frontend
npm install
```

## Optional installation (bun)

```bash
cd src/frontend
bun install
```

## Running the frontend

Development server:

```bash
cd src/frontend
npm run dev
```

App URL: `http://localhost:3000/chat`

## Backend integration

The chat UI currently calls backend APIs directly via:

```ts
const BACKEND_URL = "http://localhost:8013"
```

in `components/chat/chat-interface.tsx`.

So backend must be running on port `8013` for chat/session/voice features to work, or the port for the backend can be changed.

### Backend endpoints used by frontend

| Method | Endpoint |
|---|---|
| `GET` | `/sessions/patient101` |
| `GET` | `/sessions/{session_id}/messages` |
| `POST` | `/sessions/patient101` |
| `POST` | `/sessions/{session_id}/message` |
| `DELETE` | `/gdpr/sessions/{session_id}` |
| `POST` | `/transcribe` |

## Notes

1. The chat currently uses fixed patient ID `patient101` in frontend requests.
2. `next.config.ts` contains a rewrite from `/api/python/:path*` to backend `127.0.0.1:8013`, but current chat code uses direct backend URLs.
3. No frontend `.env` variables are required by current code.
