"""
Audio Quality Analysis Tools
For analyzing and benchmarking generated audio quality
"""

import numpy as np
import soundfile as sf
import librosa
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """Analyze audio quality metrics"""

    def __init__(self):
        self.metrics = {}

    def analyze_file(self, file_path: str) -> Dict:
        """
        Perform comprehensive audio analysis

        Args:
            file_path: Path to audio file

        Returns:
            Dict with analysis results
        """
        logger.info(f"Analyzing: {file_path}")

        # Load audio
        audio, sr = librosa.load(file_path, sr=None, mono=False)
        duration = audio.shape[-1] / sr

        # Basic info
        result = {
            'file_path': file_path,
            'duration': duration,
            'sample_rate': sr,
            'channels': 1 if audio.ndim == 1 else audio.shape[0]
        }

        # Convert to mono for analysis
        if audio.ndim > 1:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio

        # Dynamic range
        result['dynamic_range'] = self._calculate_dynamic_range(audio_mono)

        # Peak amplitude
        result['peak_amplitude'] = float(np.max(np.abs(audio_mono)))
        result['peak_db'] = self._amplitude_to_db(result['peak_amplitude'])

        # RMS level
        result['rms'] = float(np.sqrt(np.mean(audio_mono ** 2)))
        result['rms_db'] = self._amplitude_to_db(result['rms'])

        # Frequency analysis
        result['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(y=audio_mono, sr=sr)))
        result['spectral_bandwidth'] = float(np.mean(librosa.feature.spectral_bandwidth(y=audio_mono, sr=sr)))

        # Zero crossing rate (brightness indicator)
        result['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(audio_mono)))

        # Tempo detection
        try:
            tempo, _ = librosa.beat.beat_track(y=audio_mono, sr=sr)
            result['estimated_tempo'] = float(tempo)
        except:
            result['estimated_tempo'] = None

        # Clipping detection
        clipping_samples = np.sum(np.abs(audio_mono) > 0.99)
        result['clipping_percentage'] = (clipping_samples / len(audio_mono)) * 100

        # Frequency range
        freqs = np.fft.fftfreq(len(audio_mono), 1/sr)
        fft = np.fft.fft(audio_mono)
        magnitude = np.abs(fft)

        # Find frequency range with significant energy
        significant = magnitude > np.max(magnitude) * 0.01
        significant_freqs = freqs[significant]
        result['freq_range_min'] = float(np.min(significant_freqs[significant_freqs > 0]))
        result['freq_range_max'] = float(np.max(significant_freqs))

        # Quality score (0-100)
        result['quality_score'] = self._calculate_quality_score(result)

        return result

    def _calculate_dynamic_range(self, audio: np.ndarray) -> float:
        """Calculate dynamic range in dB"""
        # Using 20th percentile as floor
        floor = np.percentile(np.abs(audio), 20)
        peak = np.max(np.abs(audio))

        if floor > 0:
            return 20 * np.log10(peak / floor)
        return 0

    def _amplitude_to_db(self, amplitude: float) -> float:
        """Convert linear amplitude to dB"""
        if amplitude > 0:
            return 20 * np.log10(amplitude)
        return -100

    def _calculate_quality_score(self, metrics: Dict) -> float:
        """Calculate overall quality score (0-100)"""
        score = 100

        # Penalize heavy clipping
        if metrics['clipping_percentage'] > 1:
            score -= 20
        elif metrics['clipping_percentage'] > 0.1:
            score -= 10

        # Penalize low dynamic range
        if metrics['dynamic_range'] < 6:
            score -= 15
        elif metrics['dynamic_range'] < 10:
            score -= 5

        # Penalize too quiet
        if metrics['rms_db'] < -20:
            score -= 10
        elif metrics['rms_db'] < -15:
            score -= 5

        # Penalize clipping (peak too high)
        if metrics['peak_db'] > -1:
            score -= 10
        elif metrics['peak_db'] > -3:
            score -= 5

        return max(0, min(100, score))

    def compare_tracks(self, reference_path: str, generated_path: str) -> Dict:
        """
        Compare generated track against reference

        Args:
            reference_path: Reference track path
            generated_path: Generated track path

        Returns:
            Comparison metrics
        """
        ref_analysis = self.analyze_file(reference_path)
        gen_analysis = self.analyze_file(generated_path)

        comparison = {
            'reference': ref_analysis,
            'generated': gen_analysis,
            'differences': {}
        }

        # Compare key metrics
        for key in ['dynamic_range', 'rms_db', 'spectral_centroid', 'estimated_tempo']:
            if key in ref_analysis and key in gen_analysis:
                ref_val = ref_analysis[key]
                gen_val = gen_analysis[key]
                if ref_val and gen_val:
                    diff = abs(gen_val - ref_val)
                    comparison['differences'][key] = {
                        'reference': ref_val,
                        'generated': gen_val,
                        'difference': diff,
                        'percent_diff': (diff / ref_val * 100) if ref_val != 0 else 0
                    }

        return comparison

    def print_report(self, analysis: Dict):
        """Print formatted analysis report"""
        print("\n" + "=" * 60)
        print(f"AUDIO ANALYSIS: {analysis['file_path']}")
        print("=" * 60)

        print(f"\nBasic Info:")
        print(f"  Duration:       {analysis['duration']:.2f} seconds")
        print(f"  Sample Rate:    {analysis['sample_rate']} Hz")
        print(f"  Channels:       {analysis['channels']}")

        print(f"\nLevels:")
        print(f"  Peak:           {analysis['peak_amplitude']:.4f} ({analysis['peak_db']:.2f} dB)")
        print(f"  RMS:            {analysis['rms']:.4f} ({analysis['rms_db']:.2f} dB)")
        print(f"  Dynamic Range:  {analysis['dynamic_range']:.2f} dB")

        print(f"\nFrequency:")
        print(f"  Range:          {analysis['freq_range_min']:.0f} - {analysis['freq_range_max']:.0f} Hz")
        print(f"  Spectral Cent.: {analysis['spectral_centroid']:.1f} Hz")
        print(f"  ZCR:            {analysis['zero_crossing_rate']:.4f}")

        if analysis['estimated_tempo']:
            print(f"  Estimated BPM:  {analysis['estimated_tempo']:.1f}")

        print(f"\nQuality:")
        print(f"  Clipping:       {analysis['clipping_percentage']:.2f}%")
        print(f"  Quality Score:  {analysis['quality_score']:.0f}/100")

        # Quality assessment
        score = analysis['quality_score']
        if score >= 80:
            print(f"  Assessment:    EXCELLENT")
        elif score >= 60:
            print(f"  Assessment:    GOOD")
        elif score >= 40:
            print(f"  Assessment:    FAIR")
        else:
            print(f"  Assessment:    POOR")

        print("=" * 60)


def benchmark_generation_time(func, *args, **kwargs):
    """
    Benchmark generation time for a function

    Returns:
        Tuple of (result, elapsed_time)
    """
    import time

    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start

    return result, elapsed


# Standalone test
def test_analyzer():
    """Test audio analyzer"""
    import tempfile
    import numpy as np

    analyzer = AudioAnalyzer()

    # Create test audio
    sr = 44100
    duration = 5
    t = np.linspace(0, duration, int(sr * duration))

    # Mix of frequencies
    audio = (
        np.sin(2 * np.pi * 261.63 * t) * 0.3 +  # C4
        np.sin(2 * np.pi * 523.25 * t) * 0.15 +  # C5
        np.sin(2 * np.pi * 783.99 * t) * 0.1      # G5
    )

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio, sr)
        analysis = analyzer.analyze_file(f.name)
        analyzer.print_report(analysis)


if __name__ == "__main__":
    test_analyzer()
