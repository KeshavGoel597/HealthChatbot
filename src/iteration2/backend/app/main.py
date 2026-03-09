from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from app.routers import chat, sessions
from app.routers import gdpr_router
from app.routers.sessions import run_retention_cleanup

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Robert — AI Medical Assistant Backend",
    description=(
        "Backend for the Robert AI Medical Assistant (Iteration 2). "
        "GDPR-compliant: implements Art. 5, 15, 16, 17, and 22 of the GDPR."
    ),
    version="0.3.0"
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


@app.on_event("startup")
async def startup_event():
    """
    GDPR Art. 5(1)(e) — Storage Limitation.
    On every backend startup, scan all stored sessions and delete those
    that have passed their expires_at retention deadline (default: 30 days).
    """
    print("[STARTUP] Running GDPR retention cleanup...")
    run_retention_cleanup()


@app.get("/")
async def root():
    return {
        "message": "Robert AI Medical Assistant Backend — running",
        "gdpr_compliance": "Art. 5, 15, 16, 17, 22",
        "context_compaction": "enabled"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8013)
