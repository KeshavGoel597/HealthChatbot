# `src` Workspace README

This `src/` folder contains the backend, frontend, data artifacts, and local dev environment files.

## Files and folders in this directory

| Path | What it is |
|---|---|
| `backend/` | FastAPI backend service (RAG pipeline, chat/session/GDPR APIs). |
| `frontend/` | Next.js frontend application. |
| `GraphModel_SNOMED_CUI_Embedding.pkl` | See explanation in `./backend/README.md` (Required data files section). |
| `SNOMED_CUI_MAJID_Graph_wSelf.pkl` | See explanation in `./backend/README.md` (Required data files section). |
| `sm_t047_cui_aui_eng.pkl` | See explanation in `./backend/README.md` (Required data files section). |
| `flake.nix` | Nix flake definition for reproducible dev shell/tooling. |
| `flake.lock` | Locked dependency versions for the Nix flake. |
| `.envrc` | Direnv hook (`use flake`) to auto-enter the Nix dev shell. |
| `.gitignore` | Ignore rules for this `src/` workspace. |

## Quick navigation

- [Backend](./backend/README.md)
- [Frontend](./frontend/README.md)
