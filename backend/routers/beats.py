"""
Brazilian Phonk Beat Generator Router

Dedicated endpoint for generating Phonk beats without requiring audio input.
Uses Bark AI to generate authentic Brazilian Phonk (Phonk Brasileiro) beats.
"""

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import yaml
import json
import os
import numpy as np
from typing import Dict, Any

# Import Bark generator
try:
    from modules.bark_generator import BarkGenerator
    BARK_AVAILABLE = True
except ImportError as e:
    BARK_AVAILABLE = False
    print(f"WARNING: Bark not available: {e}")

# Import audio processing utilities
from utils.audio_processing import (
    crossfade_chunks,
    apply_phonk_effects,
    save_audio
)
from utils.file_manager import generate_job_id, get_output_path

router = APIRouter(tags=["beats"])

# Load Brazilian Phonk presets
with open("config/brazilian_phonk_presets.json") as f:
    BRAZILIAN_PHONK_PRESETS = json.load(f)["presets"]

# Create preset lookup by ID
PRESETS_BY_ID: Dict[str, Dict] = {p["id"]: p for p in BRAZILIAN_PHONK_PRESETS}


class BeatResponse(BaseModel):
    """Response model for beat generation endpoint"""
    job_id: str
    beat_url: str
    status: str
    preset: Dict[str, Any]
    duration: int


@router.get("/presets")
async def get_presets():
    """Get available Brazilian Phonk presets"""
    return {
        "presets": BRAZILIAN_PHONK_PRESETS,
        "count": len(BRAZILIAN_PHONK_PRESETS)
    }


@router.post("/generate")
async def generate_beat(
    style: str = Form(..., description="Style preset ID (e.g., 'montagem-batidao')")
):
    """
    Generate Brazilian Phonk beat without audio input

    Pipeline:
    1. Load style preset
    2. Generate audio chunks with Bark (30s each)
    3. Crossfade chunks into full beat (60-120s)
    4. Apply Phonk effects (distortion, compression, bass boost)
    5. Save and return beat URL

    Target duration: 60-120 seconds
    """
    if not BARK_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Bark generator not available. Install with: pip install git+https://github.com/suno-ai/bark.git"
        )

    # Validate preset
    preset = PRESETS_BY_ID.get(style)
    if not preset:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid style preset: {style}. Available: {list(PRESETS_BY_ID.keys())}"
        )

    job_id = generate_job_id()
    target_duration = preset["duration"]

    try:
        print(f"\n=== Generating Brazilian Phonk Beat ===")
        print(f"Job ID: {job_id}")
        print(f"Style: {preset['name']}")
        print(f"Duration: {target_duration}s")
        print(f"Prompt: {preset['prompt'][:100]}...")
        print(f"=====================================\n")

        # Initialize Bark generator
        bark_gen = BarkGenerator()

        # Calculate number of chunks needed (Bark generates ~12s per call)
        # We'll use the prompt directly with generate_music to get more control
        chunk_duration = 12  # seconds per Bark generation (actual is ~10-13s)
        num_chunks = int(np.ceil(target_duration / chunk_duration))

        chunks = []
        for i in range(num_chunks):
            chunk_num = i + 1
            print(f"Generating chunk {chunk_num}/{num_chunks}...")

            # Generate audio with Bark using the preset's prompt
            audio, sr = bark_gen.generate_music(
                prompt=preset["prompt"],
                temperature=preset["temperature"]
            )

            chunks.append(audio)
            print(f"Chunk {chunk_num} complete: {len(audio)/sr:.1f}s")

        print("\nCrossfading chunks...")
        # Crossfade all chunks into one continuous beat
        full_beat = crossfade_chunks(chunks, crossfade_ms=500, sample_rate=sr)

        # Trim to exact duration if needed
        target_samples = int(target_duration * sr)
        if len(full_beat) > target_samples:
            full_beat = full_beat[:target_samples]

        print(f"Combined beat: {len(full_beat)/sr:.1f}s")

        print("Applying Phonk effects...")
        # Apply Phonk effects (distortion, compression, bass boost)
        processed = apply_phonk_effects(full_beat, preset, sample_rate=sr)

        # Save beat with upsampling to 48kHz for better quality
        # This significantly improves audio clarity and reduces the "old radio" sound
        output_path = get_output_path("beats", job_id)
        save_audio(processed, output_path, sample_rate=sr, target_sample_rate=48000)

        print(f"\n=== Beat Generation Complete ===")
        print(f"Saved to: {output_path}")
        print(f"Duration: {len(processed)/sr:.1f}s")
        print(f"Style: {preset['name']}")
        print("==================================\n")

        return BeatResponse(
            job_id=job_id,
            beat_url=f"/outputs/beats/{job_id}.wav",
            status="completed",
            preset=preset,
            duration=int(len(processed)/sr)
        )

    except Exception as e:
        print(f"\nERROR: Beat generation failed: {e}")
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Beat generation failed: {str(e)}"
        )


@router.get("/status/{job_id}")
async def check_beat_status(job_id: str):
    """Check generation status for a beat"""
    beat_path = get_output_path("beats", f"{job_id}.wav")

    from utils.file_manager import file_exists
    if file_exists(beat_path):
        return {
            "job_id": job_id,
            "status": "completed",
            "beat_url": f"/outputs/beats/{job_id}.wav"
        }
    else:
        return {
            "job_id": job_id,
            "status": "processing"
        }


@router.get("/download/{job_id}")
async def download_beat(job_id: str):
    """Download generated beat file"""
    beat_path = get_output_path("beats", f"{job_id}.wav")

    from utils.file_manager import file_exists
    if not file_exists(beat_path):
        raise HTTPException(
            status_code=404,
            detail=f"Beat not found: {job_id}"
        )

    return FileResponse(
        path=beat_path,
        filename=f"brazilian-phonk-{job_id}.wav",
        media_type="audio/wav"
    )
