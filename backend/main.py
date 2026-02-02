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
    "outputs/final",
    "outputs/beats"
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
    from utils.audio_utils import check_gpu_availability
    gpu_info = check_gpu_availability()

    return {
        "status": "healthy",
        "service": "Dream Architect",
        "version": "1.0.0",
        "gpu_available": gpu_info["available"],
        "generate_router_available": GENERATE_ROUTER_AVAILABLE
    }

# Include routers (with graceful fallback)
try:
    from routers import generate
    app.include_router(generate.router, prefix="/api", tags=["generation"])
    GENERATE_ROUTER_AVAILABLE = True
except ImportError as e:
    GENERATE_ROUTER_AVAILABLE = False
    print(f"Warning: Generate router not available: {e}")
    print("Some ML dependencies may be missing. Install with: pip install crepe audiocraft openai")

try:
    from routers import beats
    app.include_router(beats.router, prefix="/api/beats", tags=["beats"])
    BEATS_ROUTER_AVAILABLE = True
except ImportError as e:
    BEATS_ROUTER_AVAILABLE = False
    print(f"Warning: Beats router not available: {e}")

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
