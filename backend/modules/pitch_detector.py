"""
Pitch Detection Module using CREPE
Optimized for 4GB VRAM usage with 'small' model capacity

Extracts pitch values from hummed audio and converts them to time-stamped data.
"""

import crepe
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import torch
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PitchDetector:
    """
    Pitch detection using CREPE (Convolutional Representation for Pitch Estimation)

    Optimized for low VRAM usage:
    - Uses 'small' model capacity (~500MB VRAM)
    - CPU fallback available
    - Efficient audio preprocessing
    """

    def __init__(
        self,
        model_capacity: str = 'small',
        device: Optional[str] = None
    ):
        """
        Initialize pitch detector

        Args:
            model_capacity: CREPE model size ('tiny', 'small', 'medium', 'large', 'full')
                           'small' recommended for 4GB VRAM target
            device: Force device ('cuda' or 'cpu'), auto-detect if None
        """
        self.model_capacity = model_capacity
        self.device = self._determine_device(device)
        self.model_loaded = False

        logger.info(f"PitchDetector initialized with model_capacity='{model_capacity}', device='{self.device}'")

    def _determine_device(self, device: Optional[str]) -> str:
        """Determine the best device for computation"""
        if device:
            return device

        if torch.cuda.is_available():
            # Check available memory
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if total_memory_gb >= 4:
                return 'cuda'
            else:
                logger.warning(f"GPU has only {total_memory_gb:.1f}GB, using CPU for stability")
                return 'cpu'

        return 'cpu'

    def _preprocess_audio(
        self,
        audio: np.ndarray,
        sr: int,
        target_sr: int = 16000
    ) -> np.ndarray:
        """
        Preprocess audio for pitch detection

        Args:
            audio: Input audio array
            sr: Sample rate of input audio
            target_sr: Target sample rate (CREPE works best at 16kHz)

        Returns:
            Preprocessed audio array
        """
        # Resample to 16kHz (CREPE requirement)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # High-pass filter to remove low frequency rumble
        # Simple implementation using librosa
        audio = librosa.effects.harmonic(audio, margin=8)

        # Normalize to [-1, 1]
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        return audio

    def extract_pitches(
        self,
        audio_file_path: str,
        min_confidence: float = 0.5,
        viterbi: bool = True,
        step_size: int = 10
    ) -> Dict:
        """
        Extract pitches from audio file

        Args:
            audio_file_path: Path to audio file
            min_confidence: Minimum confidence threshold (0-1)
            viterbi: Use Viterbi algorithm for smoother pitch
            step_size: Step size in milliseconds

        Returns:
            Dictionary with pitch data including times, frequencies, confidences
        """
        logger.info(f"Extracting pitches from: {audio_file_path}")

        # Load audio
        audio, sr = librosa.load(audio_file_path, sr=None, mono=True)
        duration = len(audio) / sr

        if duration < 0.5:
            raise ValueError(f"Audio too short: {duration:.2f}s (minimum 0.5s)")

        # Preprocess
        audio_processed = self._preprocess_audio(audio, sr, target_sr=16000)

        # Run CREPE pitch detection
        logger.info(f"Running CREPE (model_capacity={self.model_capacity})...")

        time_stamps, frequencies, confidences, activation = crepe.predict(
            audio_processed,
            sr=16000,
            model_capacity=self.model_capacity,
            viterbi=viterbi,
            step_size=step_size,
            center=True,
            pad=True
        )

        # Smooth and clean the pitch contour
        frequencies_smoothed = self.smooth_pitches(
            frequencies,
            confidences,
            min_confidence=min_confidence
        )

        # Calculate statistics
        valid_pitches = frequencies_smoothed[frequencies_smoothed > 0]
        pitch_range = (
            float(np.min(valid_pitches)) if len(valid_pitches) > 0 else 0,
            float(np.max(valid_pitches)) if len(valid_pitches) > 0 else 0
        )

        result = {
            "times": time_stamps.tolist(),
            "pitches": frequencies_smoothed.tolist(),
            "confidences": confidences.tolist(),
            "avg_confidence": float(np.mean(confidences)),
            "pitch_range": pitch_range,
            "duration": duration,
            "sample_rate": 16000,
            "model_capacity": self.model_capacity
        }

        logger.info(
            f"Pitch extraction complete: "
            f"{len(valid_pitches)} valid pitches, "
            f"range: {pitch_range[0]:.1f}-{pitch_range[1]:.1f} Hz"
        )

        return result

    def smooth_pitches(
        self,
        pitches: np.ndarray,
        confidences: np.ndarray,
        min_confidence: float = 0.5
    ) -> np.ndarray:
        """
        Smooth pitch contour and remove outliers

        Args:
            pitches: Raw pitch array
            confidences: Confidence array
            min_confidence: Minimum confidence threshold

        Returns:
            Smoothed pitch array
        """
        pitches_smoothed = pitches.copy()

        # Set low confidence pitches to 0 (unvoiced)
        low_confidence_mask = confidences < min_confidence
        pitches_smoothed[low_confidence_mask] = 0

        # Median filter to smooth the contour
        from scipy.ndimage import median_filter
        pitches_smoothed = median_filter(pitches_smoothed, size=3)

        # Remove octave jumps
        for i in range(1, len(pitches_smoothed)):
            if pitches_smoothed[i] > 0 and pitches_smoothed[i-1] > 0:
                ratio = pitches_smoothed[i] / pitches_smoothed[i-1]

                # Check for octave jump (2x or 0.5x)
                if ratio > 1.8:  # Likely octave up jump
                    pitches_smoothed[i] = pitches_smoothed[i] / 2
                elif ratio < 0.55:  # Likely octave down jump
                    pitches_smoothed[i] = pitches_smoothed[i] * 2

        return pitches_smoothed

    def visualize_pitches(
        self,
        pitches: np.ndarray,
        times: np.ndarray,
        confidences: np.ndarray,
        output_path: Optional[str] = None
    ):
        """
        Create pitch visualization plot

        Args:
            pitches: Pitch array
            times: Time array
            confidences: Confidence array
            output_path: Path to save plot (optional)
        """
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # Plot pitch contour
        ax1.plot(times, pitches, 'b-', linewidth=2)
        ax1.set_ylabel('Pitch (Hz)')
        ax1.set_title('Pitch Contour')
        ax1.grid(True, alpha=0.3)

        # Plot confidence
        ax2.plot(times, confidences, 'r-', linewidth=2)
        ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Min Confidence')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Confidence')
        ax2.set_title('Pitch Detection Confidence')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved pitch visualization to: {output_path}")
        else:
            plt.show()

        plt.close()

    def get_voiced_segments(
        self,
        pitches: np.ndarray,
        confidences: np.ndarray,
        min_confidence: float = 0.7,
        min_duration: float = 0.1
    ) -> list:
        """
        Extract continuous voiced segments from pitch data

        Args:
            pitches: Pitch array
            confidences: Confidence array
            min_confidence: Minimum confidence threshold
            min_duration: Minimum segment duration in seconds

        Returns:
            List of (start_time, end_time, avg_pitch) tuples
        """
        step_size = 0.01  # CREPE default step size
        voiced_mask = (pitches > 0) & (confidences >= min_confidence)

        segments = []
        current_segment = None

        for i, is_voiced in enumerate(voiced_mask):
            time = i * step_size

            if is_voiced:
                if current_segment is None:
                    current_segment = {
                        'start': time,
                        'pitches': [pitches[i]],
                        'indices': [i]
                    }
                else:
                    current_segment['pitches'].append(pitches[i])
                    current_segment['indices'].append(i)
            else:
                if current_segment is not None:
                    duration = time - current_segment['start']
                    if duration >= min_duration:
                        avg_pitch = np.mean(current_segment['pitches'])
                        segments.append((
                            current_segment['start'],
                            time,
                            avg_pitch
                        ))
                    current_segment = None

        # Handle final segment
        if current_segment is not None:
            duration = (len(voiced_mask) * step_size) - current_segment['start']
            if duration >= min_duration:
                avg_pitch = np.mean(current_segment['pitches'])
                segments.append((
                    current_segment['start'],
                    len(voiced_mask) * step_size,
                    avg_pitch
                ))

        return segments


# Standalone test function
def test_pitch_detector():
    """Test the pitch detector with a sample audio file"""
    detector = PitchDetector(model_capacity='small')

    # Create a test tone (440 Hz = A4)
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    test_audio = np.sin(2 * np.pi * 440 * t)

    # Save test file
    test_path = "test_tone_440hz.wav"
    sf.write(test_path, test_audio, sr)

    # Extract pitches
    result = detector.extract_pitches(test_path)

    print(f"Average pitch: {np.mean(result['pitches'][result['pitches'] > 0]):.1f} Hz")
    print(f"Expected: ~440 Hz")

    # Cleanup
    Path(test_path).unlink(missing_ok=True)


if __name__ == "__main__":
    test_pitch_detector()
