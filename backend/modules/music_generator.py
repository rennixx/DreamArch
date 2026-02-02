"""
Music Generator Module using Meta's MusicGen
Optimized for 4GB VRAM with MusicGen Small + fp16 precision

Generates melody-conditioned instrumental tracks from MIDI input.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from audiocraft.models import MusicGen
    from audiocraft.data.audio import audio_write
    AUDIOCRAFT_AVAILABLE = True
except ImportError:
    logger.warning("audiocraft not available. Install with: pip install audiocraft")
    AUDIOCRAFT_AVAILABLE = False

import pretty_midi


class MusicGenerator:
    """
    Music generation using MusicGen (optimized for 4GB VRAM)

    Features:
    - Melody-conditioned generation
    - fp16 precision for memory efficiency
    - GPU/CPU fallback
    - Model caching
    - Progress tracking
    """

    def __init__(
        self,
        model_name: str = "facebook/musicgen-small",
        device: Optional[str] = None,
        use_fp16: bool = True
    ):
        """
        Initialize music generator

        Args:
            model_name: HuggingFace model name
                       - 'facebook/musicgen-small' (~1.5GB VRAM)
                       - 'facebook/musicgen-medium' (~3GB VRAM)
                       - 'facebook/musicgen-large' (~6GB VRAM - not recommended)
            device: 'cuda', 'cpu', or None for auto-detect
            use_fp16: Use half precision for memory savings
        """
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.device = self._determine_device(device)
        self.model = None
        self.model_loaded = False
        self.available = AUDIOCRAFT_AVAILABLE

        if not self.available:
            logger.warning(
                "MusicGenerator initialized but audiocraft is not available. "
                "Install with: pip install audiocraft"
            )
        else:
            logger.info(
                f"MusicGenerator initialized: "
                f"model={model_name}, device={self.device}, fp16={use_fp16}"
            )

    @property
    def is_available(self) -> bool:
        """Check if MusicGen is available for use"""
        return self.available

    def _determine_device(self, device: Optional[str]) -> str:
        """Determine the best device for computation"""
        if device:
            return device

        if torch.cuda.is_available():
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

            # Use CUDA if we have at least 3GB (MusicGen small needs ~1.5GB)
            if total_memory_gb >= 3:
                return 'cuda'
            else:
                logger.warning(
                    f"GPU has only {total_memory_gb:.1f}GB, "
                    f"using CPU for stability"
                )
                return 'cpu'

        return 'cpu'

    def _load_model(self):
        """Load MusicGen model (lazy loading)"""
        if self.model_loaded:
            return

        if not self.available:
            raise RuntimeError(
                "audiocraft is required for music generation. "
                "Install with: pip install audiocraft"
            )

        logger.info(f"Loading MusicGen model: {self.model_name}")

        try:
            self.model = MusicGen.get_pretrained(self.model_name)

            # Set generation parameters
            self.model.set_generation_params(
                duration=45,  # Default duration, can be overridden
                temperature=1.0,
                top_k=250,
                top_p=0.9,
                cfg_coef=3.0  # Classifier-free guidance strength
            )

            # Enable fp16 if requested and on GPU
            if self.use_fp16 and self.device == 'cuda':
                # Convert model to half precision
                self.model.model = self.model.model.half()

            self.model_loaded = True
            logger.info("MusicGen model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load MusicGen model: {e}")
            raise

    def midi_to_audio(
        self,
        midi_path: str,
        output_path: Optional[str] = None,
        sample_rate: int = 32000
    ) -> Tuple[np.ndarray, int]:
        """
        Convert MIDI to simple audio for melody conditioning

        Uses sine wave synthesis to create a simple melody track.

        Args:
            midi_path: Path to MIDI file
            output_path: Optional path to save audio
            sample_rate: Output sample rate

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        midi = pretty_midi.PrettyMIDI(midi_path)
        duration = midi.get_end_time()

        # Create audio array
        audio = np.zeros(int(duration * sample_rate))

        # Synthesize each note as a sine wave
        for instrument in midi.instruments:
            for note in instrument.notes:
                # Frequency of this note
                freq = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))

                # Time indices for this note
                start_idx = int(note.start * sample_rate)
                end_idx = int(note.end * sample_rate)

                if end_idx > len(audio):
                    end_idx = len(audio)

                # Generate sine wave for this note
                t = np.arange(end_idx - start_idx) / sample_rate
                note_audio = np.sin(2 * np.pi * freq * t)

                # Apply envelope (attack, decay)
                envelope = np.ones_like(note_audio)
                attack_samples = int(0.01 * sample_rate)  # 10ms attack
                release_samples = int(0.05 * sample_rate)  # 50ms release

                if len(envelope) > attack_samples + release_samples:
                    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
                    envelope[-release_samples:] = np.linspace(1, 0, release_samples)

                note_audio = note_audio * envelope * 0.3  # Scale amplitude

                # Mix into output
                audio[start_idx:end_idx] += note_audio

        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        # Save if requested
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(
                output_path,
                torch.from_numpy(audio).unsqueeze(0),
                sample_rate
            )
            logger.info(f"Saved melody audio to: {output_path}")

        return audio, sample_rate

    def generate_from_melody(
        self,
        midi_path: str,
        style_prompt: str,
        duration: int = 45,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.9,
        progress: bool = False
    ) -> Tuple[np.ndarray, int]:
        """
        Generate music conditioned on MIDI melody

        Args:
            midi_path: Path to MIDI file for melody conditioning
            style_prompt: Text description of desired style
            duration: Output duration in seconds (max 30 for MusicGen small)
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            progress: Show generation progress

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Lazy load model
        if not self.model_loaded:
            self._load_model()

        logger.info(f"Generating music: style='{style_prompt}', duration={duration}s")

        # Convert MIDI to audio for conditioning
        melody_audio, melody_sr = self.midi_to_audio(midi_path)
        melody_tensor = torch.from_numpy(melody_audio).unsqueeze(0).float()

        # Resample if needed (MusicGen uses 32kHz)
        if melody_sr != 32000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=melody_sr,
                new_freq=32000
            )
            melody_tensor = resampler(melody_tensor)

        # Update generation parameters
        self.model.set_generation_params(
            duration=min(duration, 30),  # MusicGen small max is 30s
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        # Generate with melody conditioning
        logger.info("Starting generation...")

        try:
            with torch.no_grad():
                output = self.model.generate_with_chroma(
                    descriptions=[style_prompt],
                    melody_wavs=melody_tensor.to(self.device),
                    melody_sample_rate=32000,
                    progress=progress
                )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("GPU out of memory. Try reducing duration or use CPU.")
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "GPU out of memory. Clearing cache. "
                    "Try reducing duration or run on CPU."
                ) from e
            else:
                raise

        # Extract output
        # output shape: (batch, channels, samples)
        audio = output[0, 0].cpu().numpy()
        sample_rate = 32000

        logger.info(f"Generated {len(audio) / sample_rate:.1f}s of audio")

        return audio, sample_rate

    def generate_unconditioned(
        self,
        style_prompt: str,
        duration: int = 30,
        temperature: float = 1.0,
        progress: bool = False
    ) -> Tuple[np.ndarray, int]:
        """
        Generate music without melody conditioning

        Args:
            style_prompt: Text description of desired style
            duration: Output duration in seconds
            temperature: Sampling temperature
            progress: Show generation progress

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if not self.model_loaded:
            self._load_model()

        logger.info(f"Generating unconditioned music: style='{style_prompt}'")

        self.model.set_generation_params(
            duration=min(duration, 30),
            temperature=temperature
        )

        with torch.no_grad():
            output = self.model.generate(
                descriptions=[style_prompt],
                progress=progress
            )

        audio = output[0, 0].cpu().numpy()
        sample_rate = 32000

        return audio, sample_rate

    def save_generated_audio(
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
                audio = audio / peak

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)

        # Save
        torchaudio.save(
            output_path,
            audio_tensor,
            sr
        )

        logger.info(f"Saved audio to: {output_path}")

    def clear_cache(self):
        """Clear GPU cache to free memory"""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache")


# Standalone test
def test_music_generator():
    """Test music generator"""
    import tempfile

    # Create test MIDI
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes.append(
        pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5)
    )
    instrument.notes.append(
        pretty_midi.Note(velocity=100, pitch=64, start=0.5, end=1.0)
    )
    midi.instruments.append(instrument)

    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
        midi_path = f.name
    midi.write(midi_path)

    # Test generation (CPU only for safety)
    try:
        generator = MusicGenerator(device='cpu', use_fp16=False)

        # Generate short test
        audio, sr = generator.generate_from_melody(
            midi_path=midi_path,
            style_prompt="upbeat pop song with drums and bass",
            duration=5,  # Very short for testing
            progress=True
        )

        print(f"Generated {len(audio) / sr:.1f}s of audio")

        # Save test output
        generator.save_generated_audio(audio, sr, "test_generation.wav")
        print("Saved to: test_generation.wav")

    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        Path(midi_path).unlink(missing_ok=True)
        Path("test_generation.wav").unlink(missing_ok=True)


if __name__ == "__main__":
    test_music_generator()
