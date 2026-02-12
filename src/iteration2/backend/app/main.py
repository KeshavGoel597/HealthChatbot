from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from app.routers import chat, sessions

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Gemini Medical Chatbot Backend",
    description="Backend for the Gemini Medical Chatbot (Iteration 2)",
    version="0.2.0"
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
