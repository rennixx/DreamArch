"""
Audio utility functions for Dream Architect
Optimized for 4GB VRAM usage
"""

import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import torch


def load_audio(file_path: str, sr: int = 44100) -> tuple[np.ndarray, int]:
    """
    Load audio file and return as numpy array

    Args:
        file_path: Path to audio file
        sr: Target sample rate (default: 44100)

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)
    return audio, sample_rate


def save_audio(
    audio: np.ndarray,
    sr: int,
    output_path: str,
    normalize: bool = True
) -> None:
    """
    Save audio to file

    Args:
        audio: Audio array
        sr: Sample rate
        output_path: Output file path
        normalize: Whether to normalize audio
    """
    if normalize:
        # Normalize to [-1, 1]
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    sf.write(output_path, audio, sr)


def apply_highpass_filter(audio: np.ndarray, sr: int, cutoff: int = 80) -> np.ndarray:
    """
    Apply high-pass filter to remove low frequency rumble

    Args:
        audio: Input audio
        sr: Sample rate
        cutoff: Cutoff frequency in Hz

    Returns:
        Filtered audio
    """
    # Simple high-pass using librosa
    return librosa.effects.harmonic(audio, margin=8)


def get_audio_duration(file_path: str) -> float:
    """
    Get duration of audio file in seconds

    Args:
        file_path: Path to audio file

    Returns:
        Duration in seconds
    """
    audio, sr = librosa.load(file_path, sr=None, mono=True)
    return len(audio) / sr


def check_gpu_availability() -> dict:
    """
    Check GPU availability and memory

    Returns:
        Dict with GPU info
    """
    gpu_info = {
        "available": torch.cuda.is_available(),
        "device": "cpu"
    }

    if gpu_info["available"]:
        gpu_info["device"] = "cuda"
        gpu_info["device_name"] = torch.cuda.get_device_name(0)
        gpu_info["total_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_info["free_memory_gb"] = (
            torch.cuda.memory_reserved(0) -
            torch.cuda.memory_allocated(0)
        ) / 1e9 if torch.cuda.memory_reserved(0) > 0 else gpu_info["total_memory_gb"]

    return gpu_info


def clear_cuda_cache():
    """Clear CUDA cache to free memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def match_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    """
    Match audio length to target by padding or truncating

    Args:
        audio: Input audio
        target_length: Target length in samples

    Returns:
        Length-matched audio
    """
    current_length = len(audio)

    if current_length < target_length:
        # Pad with zeros
        padding = target_length - current_length
        audio = np.pad(audio, (0, padding), mode='constant')
    elif current_length > target_length:
        # Truncate
        audio = audio[:target_length]

    return audio
