"""
Audio Mixer & Mastering Module using Pedalboard
CPU-based mixing and mastering with LUFS normalization

Optimized for quality and low resource usage.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from pedalboard import (
        Pedalboard, Compressor, Limiter, Gain,
        HighShelfFilter, LowShelfFilter, Reverb,
        Delay, Chorus, Phaser
    )
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False
    logger.warning("Pedalboard not available. Install with: pip install pedalboard")

try:
    import pyloudnorm as pyln
    PYLOUDNORM_AVAILABLE = True
except ImportError:
    PYLOUDNORM_AVAILABLE = False
    logger.warning("pyloudnorm not available. Install with: pip install pyloudnorm")


class AudioMixer:
    """
    Audio mixing and mastering using Pedalboard

    Features:
    - Multi-track mixing
    - CPU-based effects processing
    - LUFS loudness normalization
    - Mastering chain (EQ, compression, limiting)
    - Stereo enhancement
    - Multiple export formats
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        target_loudness: float = -14.0
    ):
        """
        Initialize audio mixer

        Args:
            sample_rate: Output sample rate
            target_loudness: Target LUFS for mastering (streaming standard)
        """
        self.sample_rate = sample_rate
        self.target_loudness = target_loudness

        logger.info(
            f"AudioMixer initialized: sr={sample_rate}, "
            f"target_loudness={target_loudness} LUFS"
        )

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        audio, sr = sf.read(file_path)

        # Convert to mono if stereo for processing
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)

        logger.info(f"Loaded audio: {file_path} ({audio.shape}, {sr}Hz)")

        return audio, sr

    def match_length(
        self,
        audio: np.ndarray,
        target_length: int
    ) -> np.ndarray:
        """
        Match audio length by padding or truncating

        Args:
            audio: Input audio
            target_length: Target length in samples

        Returns:
            Length-matched audio
        """
        current_length = audio.shape[0]

        if current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            if audio.ndim == 1:
                audio = np.pad(audio, (0, padding), mode='constant')
            else:
                padding_shape = [(0, 0)] * (audio.ndim - 1) + [(0, padding)]
                audio = np.pad(audio, padding_shape, mode='constant')

        elif current_length > target_length:
            # Truncate
            audio = audio[:target_length]

        return audio

    def balance_levels(
        self,
        tracks: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Auto-balance track levels using RMS

        Args:
            tracks: Dict of track_name -> audio_array

        Returns:
            Dict with balanced audio
        """
        target_rms = {
            "instrumental": 0.25,
            "vocals": 0.15,
            "solo": 0.18,
            "drums": 0.20
        }

        balanced = {}

        for track_name, audio in tracks.items():
            if audio.size == 0:
                continue

            # Calculate RMS
            rms = np.sqrt(np.mean(audio ** 2))

            # Skip if silent
            if rms < 1e-6:
                balanced[track_name] = audio
                continue

            # Get target RMS
            target = target_rms.get(track_name, 0.2)

            # Calculate scaling factor
            scale = target / rms if rms > 0 else 1.0

            # Apply scaling
            balanced[track_name] = audio * scale

            logger.info(f"Balanced {track_name}: RMS {rms:.4f} -> {target:.4f} (x{scale:.2f})")

        return balanced

    def mix_tracks(
        self,
        instrumental_path: str,
        vocal_path: Optional[str] = None,
        solo_path: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Mix multiple tracks together

        Args:
            instrumental_path: Path to instrumental track
            vocal_path: Path to vocal track (optional)
            solo_path: Path to solo track (optional)
            output_path: Path to save mix (optional)

        Returns:
            Path to mixed audio file
        """
        logger.info("Starting track mixing...")

        # Load instrumental
        inst_audio, inst_sr = self.load_audio(instrumental_path)

        # Ensure stereo
        if inst_audio.ndim == 1:
            inst_audio = np.column_stack([inst_audio, inst_audio])

        target_length = inst_audio.shape[0]

        # Initialize mix with instrumental
        mix = inst_audio.copy()

        # Add vocals if present
        if vocal_path and Path(vocal_path).exists():
            vocal_audio, _ = self.load_audio(vocal_path)

            # Match length
            vocal_audio = self.match_length(vocal_audio, target_length)

            # Ensure stereo
            if vocal_audio.ndim == 1:
                vocal_audio = np.column_stack([vocal_audio, vocal_audio])

            # Mix at reduced level
            mix = mix + vocal_audio * 0.5
            logger.info("Added vocals to mix")

        # Add solo if present
        if solo_path and Path(solo_path).exists():
            solo_audio, _ = self.load_audio(solo_path)

            # Match length
            solo_audio = self.match_length(solo_audio, target_length)

            # Ensure stereo
            if solo_audio.ndim == 1:
                solo_audio = np.column_stack([solo_audio, solo_audio])

            # Mix
            mix = mix + solo_audio * 0.7
            logger.info("Added solo to mix")

        # Prevent clipping
        peak = np.max(np.abs(mix))
        if peak > 0.95:
            mix = mix / peak * 0.95
            logger.info(f"Applied limiting: peak was {peak:.3f}")

        # Save if output path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, mix, self.sample_rate)
            logger.info(f"Mixed audio saved to: {output_path}")

        return output_path or ""

    def apply_effects_bus(
        self,
        audio: np.ndarray,
        effect_preset: str = "standard"
    ) -> np.ndarray:
        """
        Apply master bus effects

        Args:
            audio: Input audio
            effect_preset: Preset name (standard, warm, bright)

        Returns:
            Processed audio
        """
        if not PEDALBOARD_AVAILABLE:
            return audio

        board = Pedalboard()

        # EQ based on preset
        if effect_preset == "warm":
            board.append(LowShelfFilter(cutoff_frequency_hz=200, gain_db=2.0))
            board.append(HighShelfFilter(cutoff_frequency_hz=6000, gain_db=-1.0))
        elif effect_preset == "bright":
            board.append(LowShelfFilter(cutoff_frequency_hz=150, gain_db=1.0))
            board.append(HighShelfFilter(cutoff_frequency_hz=8000, gain_db=3.0))
        else:  # standard
            board.append(LowShelfFilter(cutoff_frequency_hz=100, gain_db=1.0))
            board.append(HighShelfFilter(cutoff_frequency_hz=8000, gain_db=1.5))

        # Compression
        board.append(Compressor(
            threshold_db=-18,
            ratio=2.5,
            attack_ms=10,
            release_ms=100
        ))

        # Process
        return board(audio, self.sample_rate)

    def master_track(
        self,
        input_path: str,
        output_path: str,
        target_loudness: Optional[float] = None,
        export_format: str = "mp3"
    ) -> str:
        """
        Master a track with professional processing chain

        Args:
            input_path: Path to input audio
            output_path: Path to save mastered audio
            target_loudness: Target LUFS (uses default if None)
            export_format: Output format ('mp3', 'wav', 'flac')

        Returns:
            Path to mastered audio file
        """
        logger.info(f"Mastering track: {input_path}")

        # Load audio
        audio, sr = sf.read(input_path)

        # Ensure stereo
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])

        # Apply mastering chain
        processed = self._apply_mastering_chain(audio)

        # Loudness normalization
        if PYLOUDNORM_AVAILABLE:
            processed = self._normalize_loudness(
                processed,
                target_loudness or self.target_loudness
            )
        else:
            # Fallback: simple normalization
            peak = np.max(np.abs(processed))
            if peak > 0:
                processed = processed / peak * 0.95

        # Export
        self._export_audio(
            processed,
            sr,
            output_path,
            format=export_format
        )

        logger.info(f"Mastered track saved to: {output_path}")

        return output_path

    def _apply_mastering_chain(self, audio: np.ndarray) -> np.ndarray:
        """Apply the full mastering chain"""
        if not PEDALBOARD_AVAILABLE:
            return audio

        # Create mastering board
        board = Pedalboard([
            # EQ - Tonal balance
            LowShelfFilter(cutoff_frequency_hz=100, gain_db=1.0),
            HighShelfFilter(cutoff_frequency_hz=8000, gain_db=1.5),

            # Compressor - Glue and control
            Compressor(
                threshold_db=-18,
                ratio=2.0,
                attack_ms=10,
                release_ms=100
            ),

            # Subtle reverb for depth
            Reverb(
                room_size=0.3,
                damping=0.7,
                wet_level=0.1,
                dry_level=0.9,
                width=1.0
            ),

            # Final limiter
            Limiter(threshold_db=-1.0, release_ms=50),

            # Output gain
            Gain(gain_db=-0.5)
        ])

        return board(audio, self.sample_rate)

    def _normalize_loudness(
        self,
        audio: np.ndarray,
        target_loudness: float
    ) -> np.ndarray:
        """
        Normalize audio to target LUFS

        Args:
            audio: Input audio
            target_loudness: Target LUFS

        Returns:
            Normalized audio
        """
        try:
            # Create meter
            meter = pyln.Meter(self.sample_rate)

            # Measure loudness (handles mono/stereo)
            if audio.ndim == 1:
                loudness = meter.integrated_loudness(audio)
            else:
                loudness = meter.integrated_loudness(audio)

            if loudness > -70:  # Valid measurement
                # Normalize
                normalized = pyln.normalize.loudness(
                    audio,
                    loudness,
                    target_loudness
                )

                logger.info(
                    f"Loudness: {loudness:.2f} -> {target_loudness:.2f} LUFS"
                )

                return normalized
            else:
                logger.warning(f"Invalid loudness measurement: {loudness:.2f}")
                return audio

        except Exception as e:
            logger.error(f"Loudness normalization failed: {e}")
            return audio

    def _export_audio(
        self,
        audio: np.ndarray,
        sr: int,
        output_path: str,
        format: str = "mp3"
    ):
        """
        Export audio to file

        Args:
            audio: Audio array
            sr: Sample rate
            output_path: Output file path
            format: Export format
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if format == "wav":
            sf.write(output_path, audio, sr, subtype='PCM_24')

        elif format == "flac":
            sf.write(output_path, audio, sr, subtype='PCM_24')

        elif format == "mp3":
            # Use pydub for MP3 export
            try:
                from pydub import AudioSegment

                # Convert to 16-bit for pydub
                audio_16 = (audio * 32767).astype(np.int16)

                # Create AudioSegment
                if audio.ndim == 1:
                    audio_segment = AudioSegment(
                        audio_16.tobytes(),
                        frame_rate=sr,
                        sample_width=audio_16.dtype.itemsize,
                        channels=1
                    )
                else:
                    audio_segment = AudioSegment(
                        audio_16.tobytes(),
                        frame_rate=sr,
                        sample_width=2,  # 16-bit
                        channels=audio.shape[1]
                    )

                # Export MP3
                audio_segment.export(output_path, format="mp3", bitrate="320k")

            except ImportError:
                logger.warning("pydub not available, saving as WAV instead")
                sf.write(output_path, audio, sr)

        else:
            raise ValueError(f"Unsupported format: {format}")

    def enhance_stereo_width(
        self,
        audio: np.ndarray,
        width: float = 1.2
    ) -> np.ndarray:
        """
        Enhance stereo width using mid-side processing

        Args:
            audio: Input audio (stereo)
            width: Width multiplier (>1 = wider, <1 = narrower)

        Returns:
            Width-enhanced audio
        """
        if audio.ndim == 1 or audio.shape[1] < 2:
            return audio  # Can't widen mono

        # Extract channels
        left = audio[:, 0]
        right = audio[:, 1]

        # Mid-side encoding
        mid = (left + right) / 2
        side = (left - right) / 2

        # Enhance side
        side = side * width

        # Mid-side decoding
        left_out = mid + side
        right_out = mid - side

        # Re-stack
        return np.column_stack([left_out, right_out])


# Standalone test
def test_mixer():
    """Test audio mixer"""
    import tempfile
    import numpy as np

    mixer = AudioMixer(sample_rate=44100)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test tracks
        duration = 5  # seconds
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration))

        # Instrumental: simple chords
        inst = np.sin(2 * np.pi * 261.63 * t) * 0.3  # C4
        inst += np.sin(2 * np.pi * 329.63 * t) * 0.2  # E4
        inst += np.sin(2 * np.pi * 392.00 * t) * 0.2  # G4
        inst = np.column_stack([inst, inst])

        # Vocals: melody
        vocals = np.sin(2 * np.pi * 523.25 * t) * 0.15  # C5
        vocals = np.column_stack([vocals, vocals])

        # Save test files
        inst_path = f"{tmpdir}/instrumental.wav"
        vocal_path = f"{tmpdir}/vocals.wav"

        sf.write(inst_path, inst, sr)
        sf.write(vocal_path, vocals, sr)

        # Mix
        mixed_path = f"{tmpdir}/mixed.wav"
        mixer.mix_tracks(inst_path, vocal_path, output_path=mixed_path)

        # Master
        final_path = f"{tmpdir}/final.mp3"
        mixer.master_track(mixed_path, final_path)

        print(f"Test mix complete: {final_path}")

        # Cleanup
        Path(inst_path).unlink(missing_ok=True)
        Path(vocal_path).unlink(missing_ok=True)
        Path(mixed_path).unlink(missing_ok=True)
        Path(final_path).unlink(missing_ok=True)


if __name__ == "__main__":
    test_mixer()
