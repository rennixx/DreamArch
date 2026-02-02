"""
Dream Architect - FastAPI Backend
AI Music Generation from Hummed Melodies

Optimized for 4GB VRAM, ~60s generation time
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path

# Create output directories if they don't exist
output_dirs = [
    "uploads",
    "outputs/midi",
    "outputs/instrumental",
    "outputs/vocals",
    "outputs/mixed",
    "outputs/final"
]

for dir_path in output_dirs:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Dream Architect",
    description="Turn hummed melodies into full AI-generated songs",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for outputs
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API is running"""
    return {
        "status": "healthy",
        "service": "Dream Architect",
        "version": "1.0.0",
        "gpu_available": False  # Will be updated with actual GPU check
    }

# Include routers
from routers import generate
app.include_router(generate.router, prefix="/api", tags=["generation"])

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Dream Architect API",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
