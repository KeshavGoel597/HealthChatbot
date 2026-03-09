from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from app.routers import chat, sessions
from app.services.rag.embeddings import EmbeddingIndex
from app.services.rag.graph import KnowledgeGraph
from app.services.rag.term_extractor import TermExtractor

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
    # Startup: load RAG resources once
    app.state.embedding_index = EmbeddingIndex(_EMBEDDING_PATH)
    app.state.knowledge_graph = KnowledgeGraph(_GRAPH_PATH)
    app.state.term_extractor = TermExtractor()
    yield
    # Shutdown: nothing to clean up


app = FastAPI(
    title="Gemini Medical Chatbot Backend",
    description="Backend for the Gemini Medical Chatbot (Iteration 2)",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS Configuration
origins = [
    "http://localhost:3000",
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

@app.get("/")
async def root():
    return {"message": "Gemini Medical Chatbot Backend verified running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8013)
