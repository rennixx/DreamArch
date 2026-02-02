"""
Generation router for Dream Architect
Orchestrates the full music generation pipeline
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Optional imports with error tracking
MODULES_AVAILABLE = {}
MODULES_MISSING = []

try:
    from modules.pitch_detector import PitchDetector
    MODULES_AVAILABLE['pitch_detector'] = True
except ImportError as e:
    MODULES_AVAILABLE['pitch_detector'] = False
    MODULES_MISSING.append(f"pitch_detector ({e})")

from modules.midi_processor import MIDIProcessor  # Always available

try:
    from modules.music_generator import MusicGenerator
    MODULES_AVAILABLE['music_generator'] = True
except ImportError as e:
    MODULES_AVAILABLE['music_generator'] = False
    MODULES_MISSING.append(f"music_generator ({e})")

try:
    from modules.lyrics_generator import LyricsGenerator
    MODULES_AVAILABLE['lyrics_generator'] = True
except ImportError as e:
    MODULES_AVAILABLE['lyrics_generator'] = False
    MODULES_MISSING.append(f"lyrics_generator ({e})")

try:
    from modules.vocal_synth import VocalSynthesizer
    MODULES_AVAILABLE['vocal_synth'] = True
except ImportError as e:
    MODULES_AVAILABLE['vocal_synth'] = False
    MODULES_MISSING.append(f"vocal_synth ({e})")

from modules.mixer import AudioMixer  # Always available

import yaml
import json
import os
from utils.file_manager import (
    generate_job_id, get_upload_path, get_output_path,
    ensure_directories
)
from utils.audio_utils import check_gpu_availability, clear_cuda_cache

router = APIRouter(tags=["generation"])

# Load configuration
with open("config/settings.yaml") as f:
    config = yaml.safe_load(f)

with open("config/style_presets.json") as f:
    style_presets = json.load(f)["presets"]


class GenerationResponse(BaseModel):
    """Response model for generation endpoint"""
    job_id: str
    output_url: str
    status: str
    metadata: dict | None = None


@router.get("/styles")
async def get_styles():
    """Get available style presets"""
    return {
        "styles": style_presets,
        "count": len(style_presets)
    }


@router.get("/health/gpu")
async def check_gpu():
    """Check GPU availability and status"""
    gpu_info = check_gpu_availability()
    return {
        "gpu_available": gpu_info["available"],
        "device": gpu_info["device"],
        "device_name": gpu_info.get("device_name"),
        "total_memory_gb": gpu_info.get("total_memory_gb"),
        "free_memory_gb": gpu_info.get("free_memory_gb")
    }


@router.post("/generate", response_model=GenerationResponse)
async def generate_track(
    audio: UploadFile = File(..., description="Audio file (wav, webm, mp3)"),
    style: str = Form(..., description="Style preset ID")
):
    """
    Generate a full music track from a hummed melody

    Pipeline:
    1. Upload audio
    2. Extract pitch (CREPE)
    3. Convert to MIDI
    4. Generate instrumental (MusicGen)
    5. Generate lyrics (GPT-3.5)
    6. Synthesize vocals (TTS)
    7. Mix and master
    8. Return MP3

    Target time: ~60s total on 4GB VRAM
    """

    # Check if required modules are available
    if not MODULES_AVAILABLE.get('pitch_detector', False):
        raise HTTPException(
            status_code=503,
            detail="Pitch detection module not available. Install: pip install crepe tensorflow"
        )

    # MusicGen is optional - will generate vocals-only if not available
    music_gen_available = MODULES_AVAILABLE.get('music_generator', False)
    if not music_gen_available:
        print("WARNING: MusicGen not available - will generate vocals-only track")

    job_id = generate_job_id()

    try:
        # Ensure directories exist
        ensure_directories()

        # Validate style preset
        preset = next((p for p in style_presets if p["id"] == style), None)
        if not preset:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid style preset: {style}"
            )

        # 1. Save uploaded audio
        upload_path = get_upload_path(job_id, "webm")
        with open(upload_path, "wb") as f:
            f.write(await audio.read())

        # 2. Extract pitch → MIDI (optimized CREPE small)
        pitch_detector = PitchDetector(
            model_capacity=config["pitch_detection"]["crepe_model_capacity"]
        )
        pitch_data = pitch_detector.extract_pitches(upload_path)

        # 3. Convert to MIDI
        midi_processor = MIDIProcessor(
            quantize_resolution=config["midi"]["quantize_resolution"],
            default_tempo=config["midi"]["default_tempo"]
        )
        midi_path = get_output_path("midi", job_id, "mid")
        midi_data = midi_processor.pitches_to_midi(pitch_data, midi_path)

        instrumental_path = None
        sr = 22050  # Default sample rate

        # 4. Generate instrumental (MusicGen small + fp16) - OPTIONAL
        if music_gen_available:
            music_gen = MusicGenerator(
                model_name=config["generation"]["musicgen_model"],
                use_fp16=config["generation"]["use_fp16"]
            )
            instrumental_audio, sr = music_gen.generate_from_melody(
                midi_path=midi_path,
                style_prompt=preset["prompt"],
                duration=preset.get("duration", config["generation"]["musicgen_duration"])
            )

            instrumental_path = get_output_path("instrumental", job_id, "wav")
            music_gen.save_generated_audio(instrumental_audio, sr, instrumental_path)

            # Clear GPU cache before vocals
            clear_cuda_cache()

        # 5. Generate lyrics (using configured provider)
        lyrics_config = config["lyrics"]
        provider = lyrics_config.get("provider", os.getenv("LLM_PROVIDER", "ollama"))

        # Select model based on provider
        if provider == "ollama":
            model = lyrics_config.get("ollama_model", os.getenv("OLLAMA_MODEL", "ministral-3:latest"))
        elif provider == "openai":
            model = lyrics_config.get("openai_model", "gpt-3.5-turbo")
        else:
            model = lyrics_config.get("ollama_model", "ministral-3:latest")

        lyrics_gen = LyricsGenerator(
            provider=provider,
            model=model,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        lyrics_data = lyrics_gen.generate_lyrics(
            style=preset["name"],
            mood="contemplative",
            num_lines=lyrics_config["default_num_lines"]
        )

        # 6. Generate vocals (TTS - CPU based)
        vocal_config = config["vocals"]
        tts_provider = vocal_config.get("provider", os.getenv("TTS_PROVIDER", "pyttsx3"))

        # Only require API key for openai provider
        api_key = os.getenv("OPENAI_API_KEY") if tts_provider == "openai" else None

        vocal_synth = VocalSynthesizer(
            provider=tts_provider,
            api_key=api_key
        )

        vocals_path = get_output_path("vocals", job_id, "wav")
        vocal_synth.synthesize_vocals(
            lyrics=lyrics_data["lyrics"],
            melody_midi=midi_path,
            style=style,
            output_path=vocals_path
        )

        # Apply effects
        vocals_processed_path = get_output_path("vocals", f"{job_id}_processed", "wav")
        vocal_synth.apply_vocal_effects(
            audio_path=vocals_path,
            style=preset["name"],
            output_path=vocals_processed_path
        )

        # 7. Mix everything (CPU-based effects)
        mixer = AudioMixer(sample_rate=sr)

        # If no instrumental was generated, just use vocals with some basic processing
        if instrumental_path:
            mixed_path = get_output_path("mixed", job_id, "wav")
            mixer.mix_tracks(
                instrumental_path=instrumental_path,
                vocal_path=vocals_processed_path,
                output_path=mixed_path
            )
            # Master the mixed track
            final_path = get_output_path("final", job_id, "mp3")
            mixer.master_track(
                input_path=mixed_path,
                output_path=final_path,
                target_loudness=config["mixing"]["target_loudness"]
            )
        else:
            # No instrumental - just process the vocals
            final_path = get_output_path("final", job_id, "mp3")
            mixer.master_track(
                input_path=vocals_processed_path,
                output_path=final_path,
                target_loudness=config["mixing"]["target_loudness"]
            )

        # Build metadata
        metadata = {
            "style": preset["name"],
            "key": midi_data.get("detected_key"),
            "lyrics": lyrics_data.get("lyrics"),
            "has_instrumental": instrumental_path is not None,
            "instruments": "full" if instrumental_path else "vocals-only"
        }

        if instrumental_path:
            metadata["duration"] = preset.get("duration", 45)

        return GenerationResponse(
            job_id=job_id,
            output_url=f"/outputs/final/{job_id}.mp3",
            status="completed",
            metadata=metadata
        )

    except Exception as e:
        # Clean up on error
        clear_cuda_cache()
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )


@router.get("/status/{job_id}")
async def check_status(job_id: str):
    """Check generation status for a job"""
    final_path = get_output_path("final", job_id, "mp3")

    from utils.file_manager import file_exists
    if file_exists(final_path):
        return {
            "job_id": job_id,
            "status": "completed",
            "output_url": f"/outputs/final/{job_id}.mp3"
        }
    else:
        return {
            "job_id": job_id,
            "status": "processing"
        }
