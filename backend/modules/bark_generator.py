"""
Bark Music Generator Module using Suno's Bark
Text-to-audio generation for music and sound effects

Simpler alternative to MusicGen with fewer C++ dependencies.
Generates music from text prompts without requiring MIDI conditioning.
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path
import logging
from typing import Optional, Tuple
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from bark import generate_audio, preload_models
    BARK_AVAILABLE = True
except ImportError:
    BARK_AVAILABLE = False
    logger.warning("Bark not available. Install with: pip install git+https://github.com/suno-ai/bark.git")


class BarkGenerator:
    """
    Music generation using Suno's Bark (text-to-audio)

    Features:
    - Text-to-music generation
    - No MIDI conditioning required
    - GPU/CPU fallback
    - Simple API
    - Voice/music/sound effects generation
    """

    def __init__(
        self,
        device: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize Bark generator

        Args:
            device: 'cuda' or 'cpu' (None for auto-detect)
            use_gpu: Whether to use GPU if available
        """
        if not BARK_AVAILABLE:
            raise ImportError(
                "Bark is required for music generation. "
                "Install with: pip install git+https://github.com/suno-ai/bark.git"
            )

        self.use_gpu = use_gpu
        self.device = self._determine_device(device)
        self.models_loaded = False

        logger.info(
            f"BarkGenerator initialized: device={self.device}"
        )

    def _determine_device(self, device: Optional[str]) -> str:
        """Determine the best device for computation"""
        if device:
            return device

        if self.use_gpu:
            import torch
            if torch.cuda.is_available():
                return 'cuda'

        return 'cpu'

    def load_models(self):
        """Load Bark models (can be called explicitly or happens on first use)"""
        if self.models_loaded:
            return

        logger.info("Loading Bark models...")

        try:
            # Bark will use GPU if available
            preload_models()

            self.models_loaded = True
            logger.info("Bark models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Bark models: {e}")
            raise

    def generate_music(
        self,
        prompt: str,
        output_path: Optional[str] = None,
        duration: Optional[int] = None,
        temperature: float = 0.7,
        generation_params: Optional[dict] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generate music from text prompt

        Args:
            prompt: Text description of the music to generate
            output_path: Optional path to save the audio
            duration: Target duration in seconds (approximate, Bark generates ~10-15s)
            temperature: Sampling temperature (higher = more random/creative)
            generation_params: Additional generation parameters

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Load models if not already loaded
        if not self.models_loaded:
            self.load_models()

        logger.info(f"Generating music with Bark: prompt='{prompt[:50]}...'")

        # Enhance prompt for music generation
        music_prompt = self._enhance_music_prompt(prompt)

        try:
            # Generate audio with Bark
            # Bark uses semantic and coarse prompts for better control
            audio_array = generate_audio(
                text=music_prompt,
                history_prompt=None,  # Can use voice presets if needed
                temp=temperature,
                generation_params=generation_params or {}
            )

            # Bark outputs at 24kHz
            sample_rate = 24000

            logger.info(f"Generated {len(audio_array) / sample_rate:.1f}s of audio")

            # Save if path provided
            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                sf.write(output_path, audio_array, sample_rate)
                logger.info(f"Saved audio to: {output_path}")

            return audio_array, sample_rate

        except Exception as e:
            logger.error(f"Failed to generate audio: {e}")
            raise

    def _enhance_music_prompt(self, prompt: str) -> str:
        """Enhance prompt for better music generation"""
        # Add music-specific keywords if not present
        music_keywords = [
            "music", "instrumental", "beat", "melody",
            "bass", "drums", "synthesizer", "electronic"
        ]

        has_music_keyword = any(kw in prompt.lower() for kw in music_keywords)

        if not has_music_keyword:
            # Add context for music generation
            prompt = f"Instrumental music: {prompt}"

        return prompt

    def generate_with_style(
        self,
        style: str,
        mood: str,
        output_path: Optional[str] = None,
        duration: int = 15
    ) -> Tuple[np.ndarray, int]:
        """
        Generate music with style and mood

        Args:
            style: Music style (e.g., "phonk", "electronic", "rock")
            mood: Mood descriptor (e.g., "dark", "energetic", "melancholic")
            output_path: Optional path to save audio
            duration: Target duration in seconds

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        prompt = f"{style} music, {mood} mood, instrumental, high quality"

        return self.generate_music(
            prompt=prompt,
            output_path=output_path,
            duration=duration
        )

    def generate_phonk(
        self,
        output_path: Optional[str] = None,
        variation: str = "drift"
    ) -> Tuple[np.ndarray, int]:
        """
        Generate Phonk-style music

        Args:
            output_path: Optional path to save audio
            variation: Phonk variation ("drift", "aggressive", "dark")

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        prompts = {
            "drift": "phonk drift music, cowbell melody, heavy distorted 808 bass, trap hi-hats, dark Memphis style, lofi distorted, underground hip hop instrumental",
            "aggressive": "aggressive phonk beat, heavy bass, distorted 808s, fast hi-hats, intense drums, powerful rhythm",
            "dark": "dark phonk, ominous atmosphere, deep 808 sub bass, slow tempo, haunting melody, cinematic"
        }

        prompt = prompts.get(variation, prompts["drift"])

        return self.generate_music(
            prompt=prompt,
            output_path=output_path,
            temperature=0.8  # Higher temp for more variety
        )

    def save_audio(
        self,
        audio: np.ndarray,
        sr: int,
        output_path: str,
        normalize: bool = True
    ):
        """
        Save generated audio to file

        Args:
            audio: Audio array
            sr: Sample rate
            output_path: Output file path
            normalize: Whether to normalize audio
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Normalize if requested
        if normalize:
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak * 0.95

        # Save
        sf.write(output_path, audio, sr)
        logger.info(f"Saved audio to: {output_path}")

    @property
    def is_available(self) -> bool:
        """Check if Bark is available for use"""
        return BARK_AVAILABLE

    def clear_cache(self):
        """Clear GPU cache to free memory"""
        if self.device == 'cuda':
            import torch
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache")


# Standalone test
def test_bark_generator():
    """Test Bark generator"""
    import torch

    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")

    try:
        generator = BarkGenerator(device=device)

        # Test phonk generation
        print("Generating phonk beat...")
        audio, sr = generator.generate_phonk(
            output_path="test_bark_phonk.wav",
            variation="drift"
        )

        print(f"Generated {len(audio) / sr:.1f}s of phonk audio")
        print("Saved to: test_bark_phonk.wav")

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_bark_generator()
