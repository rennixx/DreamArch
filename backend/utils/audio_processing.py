"""
Audio processing utilities for Brazilian Phonk beat generation

Handles:
- Crossfading audio chunks for seamless loops
- Applying Phonk-specific effects (distortion, compression, bass boost)
- Audio manipulation and processing
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import scipy.signal as signal

logger = logging.getLogger(__name__)

# Check if Pedalboard is available
try:
    from pedalboard import Pedalboard, Distortion, Compressor, LowShelfFilter, HighShelfFilter, Reverb, Gain, PeakLimiter
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False
    logger.warning("Pedalboard not available. Install with: pip install pedalboard")


def crossfade_chunks(chunks: List[np.ndarray], crossfade_ms: int = 500, sample_rate: int = 24000) -> np.ndarray:
    """
    Combine multiple audio chunks with smooth crossfades

    Args:
        chunks: List of audio arrays (all same sample rate)
        crossfade_ms: Duration of crossfade between chunks in milliseconds
        sample_rate: Sample rate of audio

    Returns:
        Combined audio array with crossfaded transitions
    """
    if not chunks:
        raise ValueError("No chunks provided")

    if len(chunks) == 1:
        return chunks[0]

    crossfade_samples = int(crossfade_ms * sample_rate / 1000)

    result = chunks[0].copy()

    for chunk in chunks[1:]:
        # Ensure chunk is long enough for crossfade
        if len(chunk) < crossfade_samples:
            logger.warning(f"Chunk too short for crossfade ({len(chunk)} < {crossfade_samples}), appending without crossfade")
            result = np.concatenate([result, chunk])
            continue

        # Crossfade the overlap region
        overlap_end = result[-crossfade_samples:]
        chunk_start = chunk[:crossfade_samples]

        # Linear crossfade: end fades out, start fades in
        fade_out = np.linspace(1, 0, crossfade_samples)
        fade_in = np.linspace(0, 1, crossfade_samples)

        crossfaded = overlap_end * fade_out + chunk_start * fade_in

        # Replace overlap with crossfaded version
        result[-crossfade_samples:] = crossfaded

        # Append the rest of the chunk
        result = np.concatenate([result, chunk[crossfade_samples:]])

    return result


def apply_phonk_effects(audio: np.ndarray, preset: dict, sample_rate: int = 24000) -> np.ndarray:
    """
    Apply Brazilian Phonk effects chain - optimized for cleaner sound

    Effects:
    - High shelf brightness (treble enhancement for clarity)
    - Bass boost (sub-bass enhancement)
    - Light compression (glue the mix together)
    - Light distortion (optional, much reduced)
    - Peak limiter (prevent clipping)

    Args:
        audio: Input audio array
        preset: Preset dictionary with effects configuration
        sample_rate: Sample rate of audio

    Returns:
        Processed audio array
    """
    if not PEDALBOARD_AVAILABLE:
        logger.warning("Pedalboard not available, returning audio without effects")
        # Just normalize if pedalboard not available
        peak = np.max(np.abs(audio))
        if peak > 0:
            return audio / peak
        return audio

    effects = preset.get("effects", {})

    board = Pedalboard()

    # High shelf boost - brighten the audio for clarity (crucial for fixing "old radio" sound)
    # Boost frequencies above 8kHz to add presence and clarity
    board.append(HighShelfFilter(
        cutoff_frequency_hz=8000,
        gain_db=6
    ))

    # Bass boost - enhance sub-bass frequencies (80Hz and below)
    bass_boost_db = effects.get("bass_boost", 12)
    if bass_boost_db > 0:
        board.append(LowShelfFilter(
            cutoff_frequency_hz=80,
            gain_db=bass_boost_db
        ))

    # Light compression - glue the mix together (much gentler than before)
    compression = effects.get("compression", 0.5)
    if compression > 0:
        # More gentle compression settings for cleaner sound
        threshold_db = -18  # Fixed threshold for gentle compression
        ratio = 4 + compression * 8  # 4 to 12 ratio (much lower than before)

        board.append(Compressor(
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=10,
            release_ms=100
        ))

    # Light distortion - very subtle, only if requested
    distortion = effects.get("distortion", 0.3)
    if distortion > 0.4:
        # Much lower drive values for cleaner sound
        drive_db = 10 + distortion * 15  # 10-25 dB range (much lower)
        board.append(Distortion(drive_db=drive_db))

    # Peak limiter - prevent any clipping while maintaining loudness
    board.append(PeakLimiter(threshold_db=-0.5))

    # Process audio
    processed = board(audio, sample_rate)

    # Final gentle normalization
    peak = np.max(np.abs(processed))
    if peak > 0.98:
        processed = processed / peak * 0.98

    return processed


def extend_to_duration(audio: np.ndarray, target_duration: float, sample_rate: int = 24000) -> np.ndarray:
    """
    Extend audio by looping to reach target duration

    Args:
        audio: Input audio array
        target_duration: Target duration in seconds
        sample_rate: Sample rate of audio

    Returns:
        Extended audio array (looped with crossfades)
    """
    current_duration = len(audio) / sample_rate

    if current_duration >= target_duration:
        return audio

    # Calculate how many loops we need
    repeat_factor = int(np.ceil(target_duration / current_duration))

    if repeat_factor == 1:
        return audio

    # Create repeated chunks
    chunks = [audio] * repeat_factor

    # Crossfade together
    extended = crossfade_chunks(chunks, crossfade_ms=500, sample_rate=sample_rate)

    # Trim to exact duration
    target_samples = int(target_duration * sample_rate)
    if len(extended) > target_samples:
        extended = extended[:target_samples]

    return extended


def loop_with_variation(audio: np.ndarray, num_repeats: int = 2, variation: bool = False) -> np.ndarray:
    """
    Create a loop with optional variation for each repeat

    Args:
        audio: Input audio array
        num_repeats: Number of times to repeat
        variation: Whether to add slight variation to each repeat (pitch shift, etc.)

    Returns:
        Looped audio array
    """
    chunks = [audio]

    if variation and num_repeats > 1:
        # Add slight variations to each repeat
        for i in range(1, num_repeats):
            # Slight pitch shift for variation (±1 semitone max)
            shift = 1 if i % 2 == 0 else -1

            try:
                import librosa
                varied = librosa.effects.pitch_shift(audio, sr=24000, n_steps=shift)
                chunks.append(varied)
            except Exception:
                # If pitch shift fails, just repeat original
                chunks.append(audio)
    else:
        # Simple repetition
        chunks = [audio] * num_repeats

    # Crossfade all chunks
    return crossfade_chunks(chunks, crossfade_ms=500, sample_rate=24000)


def save_audio(audio: np.ndarray, output_path: str, sample_rate: int = 24000, target_sample_rate: Optional[int] = None) -> None:
    """
    Save audio to file with optional upsampling for better quality

    Args:
        audio: Audio array
        output_path: Output file path
        sample_rate: Source sample rate (default 24000 for Bark)
        target_sample_rate: Target sample rate for output (None = same as input)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Upsample if target rate is higher (improves quality significantly)
    if target_sample_rate and target_sample_rate > sample_rate:
        audio = upsample_audio(audio, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
        logger.info(f"Upsampled audio to {sample_rate}Hz for better quality")

    sf.write(output_path, audio, sample_rate)
    logger.info(f"Saved audio to: {output_path}")


def upsample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Upsample audio to higher sample rate for better quality

    Uses polyphase resampling for high-quality upsampling.
    This helps reduce the "old radio" sound from Bark's 24kHz output.

    Args:
        audio: Input audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Upsampled audio array
    """
    # Calculate the resampling ratio
    ratio = target_sr / orig_sr

    # Use polyphase resampling for high quality
    # This is better than simple linear interpolation
    number_of_samples = round(len(audio) * ratio)
    resampled = signal.resample_poly(audio, number_of_samples, len(audio))

    return resampled


def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """
    Normalize audio to target dB level

    Args:
        audio: Input audio array
        target_db: Target level in dB (typically -3 to -1 for club music)

    Returns:
        Normalized audio array
    """
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio

    # Calculate needed gain
    target_peak = 10 ** (target_db / 20)
    gain = target_peak / peak

    return audio * gain
