"""
Vocal Synthesizer Module with Multiple TTS Options
Supports: pyttsx3 (free, built-in), Edge TTS (free), OpenAI TTS (paid)

Free TTS options included - no API key required!
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TTS Options
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from pedalboard import (
        Pedalboard, Reverb, Chorus, Compressor,
        HighpassFilter, LowpassFilter, Gain, Delay
    )
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False
    logger.warning("Pedalboard not available. Install with: pip install pedalboard")


class VocalSynthesizer:
    """
    Vocal synthesis using OpenAI TTS with effects

    Features:
    - OpenAI TTS-1-HD for fast generation
    - Style-specific effects chains
    - CPU-based processing (saves GPU for MusicGen)
    - Time stretching for melody alignment
    - Multiple voice options
    """

    # Voice options mapped to different styles
    STYLE_VOICES = {
        "Pink Floyd": "onyx",      # Deep, atmospheric
        "Billie Eilish": "nova",   # Soft, intimate
        "Daft Punk": "fable",      # Smooth, robotic-friendly
        "Frank Zappa": "echo"      # Clear, articulate
    }

    # Default voice if style not found
    DEFAULT_VOICE = "alloy"

    # Style-specific effect presets
    EFFECT_PRESETS = {
        "Pink Floyd": {
            "reverb_room_size": 0.8,
            "reverb_damping": 0.5,
            "reverb_wet_level": 0.4,
            "reverb_dry_level": 0.6,
            "delay_delay_seconds": 0.3,
            "delay_feedback": 0.4,
            "delay_wet_level": 0.3,
            "chorus_rate_hz": 0.5,
            "chorus_depth": 0.3,
            "chorus_centre_delay_ms": 8,
            "highpass_cutoff": 100,
            "lowpass_cutoff": 8000
        },
        "Billie Eilish": {
            "reverb_room_size": 0.2,
            "reverb_damping": 0.9,
            "reverb_wet_level": 0.15,
            "reverb_dry_level": 0.85,
            "chorus_rate_hz": 0.2,
            "chorus_depth": 0.15,
            "chorus_centre_delay_ms": 5,
            "compressor_threshold_db": -20,
            "compressor_ratio": 4,
            "highpass_cutoff": 80,
            "lowpass_cutoff": 6000
        },
        "Daft Punk": {
            "chorus_rate_hz": 8.0,
            "chorus_depth": 0.8,
            "chorus_centre_delay_ms": 3,
            "reverb_room_size": 0.3,
            "reverb_wet_level": 0.2,
            "compressor_threshold_db": -15,
            "compressor_ratio": 6,
            "lowpass_cutoff": 4000
        },
        "Frank Zappa": {
            "reverb_room_size": 0.4,
            "reverb_wet_level": 0.2,
            "delay_delay_seconds": 0.15,
            "delay_feedback": 0.3,
            "compressor_threshold_db": -18,
            "compressor_ratio": 3,
            "highpass_cutoff": 120
        },
        "Phonk": {
            "reverb_room_size": 0.3,
            "reverb_damping": 0.7,
            "reverb_wet_level": 0.15,
            "reverb_dry_level": 0.85,
            "chorus_rate_hz": 0.3,
            "chorus_depth": 0.5,
            "chorus_centre_delay_ms": 4,
            "compressor_threshold_db": -12,
            "compressor_ratio": 8,
            "highpass_cutoff": 150,
            "lowpass_cutoff": 5000
        }
    }

    def __init__(
        self,
        provider: str = "pyttsx3",
        api_key: Optional[str] = None,
        sample_rate: int = 22050
    ):
        """
        Initialize vocal synthesizer

        Args:
            provider: TTS provider ('pyttsx3', 'edge', 'openai')
                     - pyttsx3: Free, built-in Python TTS
                     - edge: Free Microsoft Edge TTS API
                     - openai: Paid, high quality (needs API key)
            api_key: API key (only for OpenAI)
            sample_rate: Output sample rate
        """
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.sample_rate = sample_rate

        # Validate provider availability
        if provider == "pyttsx3":
            if not PYTTSX3_AVAILABLE:
                raise ImportError("pyttsx3 not available. Install: pip install pyttsx3")
            logger.info(f"VocalSynthesizer using pyttsx3 (free, built-in)")

        elif provider == "edge":
            if not REQUESTS_AVAILABLE:
                raise ImportError("requests not available. Install: pip install requests")
            logger.info(f"VocalSynthesizer using Edge TTS (free)")

        elif provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not available. Install: pip install openai")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY required for openai provider")
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"VocalSynthesizer using OpenAI TTS")

        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'pyttsx3', 'edge', or 'openai'")

    def _get_voice_for_style(self, style: str) -> str:
        """Get the appropriate voice for a given style"""
        return self.STYLE_VOICES.get(style, self.DEFAULT_VOICE)

    def _get_effects_for_style(self, style: str) -> Dict:
        """Get effects preset for a given style"""
        return self.EFFECT_PRESETS.get(style, self.EFFECT_PRESETS["Pink Floyd"])

    def synthesize_vocals(
        self,
        lyrics: List[str],
        melody_midi: str,
        style: str,
        output_path: str,
        voice: Optional[str] = None
    ) -> str:
        """
        Synthesize vocals from lyrics

        Args:
            lyrics: List of lyric lines
            melody_midi: Path to MIDI file (for timing reference)
            style: Musical style for voice selection
            output_path: Path to save vocals
            voice: Override voice selection

        Returns:
            Path to generated vocal file
        """
        if not lyrics:
            raise ValueError("Lyrics list is empty")

        logger.info(f"Synthesizing vocals: provider={self.provider}, {len(lyrics)} lines, style={style}")

        # Get voice for style
        voice = voice or self._get_voice_for_style(style)

        # Join lyrics with pauses
        lyrics_text = ". ".join(lyrics)

        try:
            if self.provider == "pyttsx3":
                return self._synthesize_with_pyttsx3(lyrics_text, output_path, voice)
            elif self.provider == "edge":
                return self._synthesize_with_edge_tts(lyrics_text, output_path)
            elif self.provider == "openai":
                return self._synthesize_with_openai(lyrics_text, voice, output_path)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

        except Exception as e:
            logger.error(f"Failed to synthesize vocals: {e}")
            raise

    def _synthesize_with_pyttsx3(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None
    ) -> str:
        """Generate speech using pyttsx3 (free, built-in)"""
        engine = pyttsx3.init()

        # Set voice properties
        voices = engine.getProperty('voices')
        if voices:
            # Try to find a good voice
            engine.setProperty('voice', voices[0].id)

        # Set rate (speed) - slightly slower for singing
        engine.setProperty('rate', 150)  # Default is 200

        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        engine.save_to_file(text, output_path)

        # CRITICAL: runAndWait() is required to actually write the file
        # pyttsx3.save_to_file() is async and only queues the operation
        engine.runAndWait()

        logger.info(f"Vocals saved to: {output_path} (pyttsx3)")
        return output_path

    def _synthesize_with_edge_tts(
        self,
        text: str,
        output_path: str
    ) -> str:
        """Generate speech using Microsoft Edge TTS (free API)"""
        # Edge TTS requires a simple GET request
        # Using the communicate API (free, no key needed)

        # Build URL for Edge TTS
        edge_url = "https://speech.platform.bing.com/consumption/tts/v1"

        # Use MP3 format and a generic voice
        params = {
            "text": text,
            "locale": "en-US",
            "gender": "Female",
            "rate": "0",  # Normal speed
            "pitch": "0",
            "volume": "0",
            "format": "audio-24khz-48kbitrate-mono-mp3"
        }

        try:
            response = requests.get(
                edge_url,
                params=params,
                timeout=30
            )
            response.raise_for_status()

            # Save audio
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"Vocals saved to: {output_path} (Edge TTS)")
            return output_path

        except requests.exceptions.RequestException as e:
            logger.error(f"Edge TTS request failed: {e}")
            # Fallback to pyttsx3
            logger.info("Falling back to pyttsx3...")
            return self._synthesize_with_pyttsx3(text, output_path, None)

    def _synthesize_with_openai(
        self,
        text: str,
        voice: str,
        output_path: str
    ) -> str:
        """Generate speech using OpenAI TTS (high quality)"""
        response = self.client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=text,
            speed=1.0,
            response_format="wav"
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(response.content)

        logger.info(f"Vocals saved to: {output_path} (OpenAI)")
        return output_path

    def apply_vocal_effects(
        self,
        audio_path: str,
        style: str,
        output_path: str,
        custom_effects: Optional[Dict] = None
    ) -> str:
        """
        Apply effects to vocals based on style

        Args:
            audio_path: Path to input vocal audio
            style: Musical style for effect selection
            output_path: Path to save processed audio
            custom_effects: Override effects

        Returns:
            Path to processed audio file
        """
        if not PEDALBOARD_AVAILABLE:
            logger.warning("Pedalboard not available, copying audio without effects")
            import shutil
            shutil.copy(audio_path, output_path)
            return output_path

        logger.info(f"Applying vocal effects: style={style}")

        # Load audio
        audio, sr = sf.read(audio_path)

        # Handle stereo
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)

        # Get effects preset
        effects = custom_effects or self._get_effects_for_style(style)

        # Build effects chain
        board = Pedalboard()

        # High-pass filter (remove rumble)
        if "highpass_cutoff" in effects:
            board.append(HighpassFilter(cutoff_frequency_hz=effects["highpass_cutoff"]))

        # Low-pass filter (warmth)
        if "lowpass_cutoff" in effects:
            board.append(LowpassFilter(cutoff_frequency_hz=effects["lowpass_cutoff"]))

        # Compressor (glue and control)
        if "compressor_threshold_db" in effects:
            board.append(Compressor(
                threshold_db=effects.get("compressor_threshold_db", -18),
                ratio=effects.get("compressor_ratio", 4),
                attack_ms=5,
                release_ms=50
            ))

        # Chorus (thickness)
        if "chorus_rate_hz" in effects:
            board.append(Chorus(
                rate_hz=effects.get("chorus_rate_hz", 0.5),
                depth=effects.get("chorus_depth", 0.3),
                centre_delay_ms=effects.get("chorus_centre_delay_ms", 8)
            ))

        # Delay (echo)
        if "delay_delay_seconds" in effects:
            board.append(Delay(
                delay_seconds=effects.get("delay_delay_seconds", 0.3),
                feedback=effects.get("delay_feedback", 0.4),
                mix=effects.get("delay_wet_level", 0.3)
            ))

        # Reverb (space)
        if "reverb_room_size" in effects:
            board.append(Reverb(
                room_size=effects.get("reverb_room_size", 0.5),
                damping=effects.get("reverb_damping", 0.5),
                wet_level=effects.get("reverb_wet_level", 0.3),
                dry_level=effects.get("reverb_dry_level", 0.7),
                width=1.0
            ))

        # Output gain (normalize)
        board.append(Gain(gain_db=-3))

        # Process
        processed = board(audio, sr)

        # Normalize to prevent clipping
        peak = np.max(np.abs(processed))
        if peak > 0.95:
            processed = processed / peak * 0.95

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, processed, sr)

        logger.info(f"Processed vocals saved to: {output_path}")

        return output_path

    def time_stretch_vocals(
        self,
        audio_path: str,
        target_duration: float,
        output_path: str
    ) -> str:
        """
        Time-stretch vocals to match target duration

        Args:
            audio_path: Path to input vocals
            target_duration: Target duration in seconds
            output_path: Path to save stretched audio

        Returns:
            Path to stretched audio file
        """
        import librosa

        # Load
        audio, sr = librosa.load(audio_path, sr=None)
        current_duration = len(audio) / sr

        if abs(current_duration - target_duration) < 0.1:
            # Already close enough
            import shutil
            shutil.copy(audio_path, output_path)
            return output_path

        # Calculate stretch ratio
        ratio = target_duration / current_duration

        # Stretch (preserve pitch)
        stretched = librosa.effects.time_stretch(audio, rate=1/ratio)

        # Save
        sf.write(output_path, stretched, sr)

        logger.info(
            f"Stretched vocals from {current_duration:.2f}s "
            f"to {target_duration:.2f}s (ratio: {ratio:.2f})"
        )

        return output_path

    def create_harmonies(
        self,
        audio_path: str,
        intervals: List[int] = [3, 5],  # Thirds and fifths
        output_dir: Optional[str] = None
    ) -> List[str]:
        """
        Create harmony tracks by pitch shifting

        Args:
            audio_path: Path to lead vocal
            intervals: List of semitone intervals for harmonies
            output_dir: Directory to save harmony tracks

        Returns:
            List of paths to harmony files
        """
        import librosa

        audio, sr = librosa.load(audio_path, sr=None)

        if output_dir is None:
            output_dir = Path(audio_path).parent
        else:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        harmony_paths = []

        for interval in intervals:
            # Pitch shift
            harmony = librosa.effects.pitch_shift(audio, sr=sr, n_steps=interval)

            # Save
            output_path = str(Path(output_dir) / f"harmony_{interval}.wav")
            sf.write(output_path, harmony, sr)
            harmony_paths.append(output_path)

            logger.info(f"Created harmony: +{interval} semitones -> {output_path}")

        return harmony_paths


# Standalone test
def test_vocal_synth():
    """Test vocal synthesizer"""
    import os
    import tempfile

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping test: OPENAI_API_KEY not set")
        return

    synth = VocalSynthesizer()

    # Test synthesis
    lyrics = [
        "Floating through the cosmic void",
        "Where time dissolves and space unfolds"
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_vocals.wav")

        try:
            # Synthesize
            synth.synthesize_vocals(
                lyrics=lyrics,
                melody_midi="",  # Not used for basic synthesis
                style="Pink Floyd",
                output_path=output_path
            )

            # Apply effects
            processed_path = os.path.join(tmpdir, "test_vocals_processed.wav")
            synth.apply_vocal_effects(
                audio_path=output_path,
                style="Pink Floyd",
                output_path=processed_path
            )

            print(f"Test vocals created: {processed_path}")

        except Exception as e:
            print(f"Test failed: {e}")


if __name__ == "__main__":
    test_vocal_synth()
