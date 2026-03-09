from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Fix OpenMP crash: multiple libomp copies (torch, sklearn, faiss) conflict
# on macOS ARM64.  Setting OMP_NUM_THREADS=1 before any native lib import
# avoids the multi-threaded fork that triggers the segfault.
os.environ.setdefault("OMP_NUM_THREADS", "1")

from app.routers import chat, sessions
from app.routers import gdpr_router

# Load environment variables
load_dotenv()

# Resolve paths to data files (relative to backend/ → project root)
_PROJECT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."
)
_EMBEDDING_PATH = os.path.join(_PROJECT_ROOT, "GraphModel_SNOMED_CUI_Embedding.pkl")
_GRAPH_PATH = os.path.join(_PROJECT_ROOT, "SNOMED_CUI_MAJID_Graph_wSelf.pkl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load RAG resources (gracefully — server works without them)
    try:
        from app.services.rag.embeddings import EmbeddingIndex
        from app.services.rag.graph import KnowledgeGraph
        from app.services.rag.term_extractor import TermExtractor

        app.state.embedding_index = EmbeddingIndex(_EMBEDDING_PATH)
        app.state.knowledge_graph = KnowledgeGraph(_GRAPH_PATH)
        app.state.term_extractor = TermExtractor()
        print("[STARTUP] RAG resources loaded successfully")
    except Exception as exc:
        print(f"[STARTUP] RAG loading failed ({exc}); server will run without RAG")
        app.state.embedding_index = None
        app.state.knowledge_graph = None
        app.state.term_extractor = None

    # GDPR Art. 5(1)(e) — Storage Limitation: clean up expired sessions
    from app.routers.sessions import run_retention_cleanup
    print("[STARTUP] Running GDPR retention cleanup...")
    run_retention_cleanup()

    yield
    # Shutdown: nothing to clean up


app = FastAPI(
    title="Robert — AI Medical Assistant Backend",
    description=(
        "Backend for the Robert AI Medical Assistant (Iteration 4). "
        "GDPR-compliant: implements Art. 5, 15, 16, 17, and 22 of the GDPR. "
        "RAG-enhanced: SNOMED knowledge graph for focused clinical context."
    ),
    version="0.4.0",
    lifespan=lifespan,
)

# CORS Configuration
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(gdpr_router.router)  # GDPR endpoints: /gdpr/...

@app.get("/")
async def root():
    return {
        "message": "Robert AI Medical Assistant Backend — running",
        "gdpr_compliance": "Art. 5, 15, 16, 17, 22",
        "context_compaction": "enabled",
        "rag_pipeline": "enabled",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8013)
