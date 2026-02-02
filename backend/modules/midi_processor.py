"""
MIDI Processor Module
Converts pitch data to quantized MIDI files with key/scale detection

Optimized for efficient processing on CPU.
"""

import pretty_midi
import numpy as np
from scipy import signal
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MIDIProcessor:
    """
    Convert pitch data to MIDI with quantization and analysis

    Features:
    - Hz to MIDI note conversion
    - Time quantization to rhythmic grid
    - Key and scale detection
    - Note boundary detection
    - Velocity variation based on confidence
    """

    # MIDI note numbers for key reference
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Major and minor key profiles for Krumhansl-Schmuckler algorithm
    MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    def __init__(
        self,
        quantize_resolution: int = 16,
        default_tempo: int = 120,
        min_note_duration: float = 0.1
    ):
        """
        Initialize MIDI processor

        Args:
            quantize_resolution: Grid resolution (4=quarter, 8=eighth, 16=sixteenth)
            default_tempo: Default tempo in BPM
            min_note_duration: Minimum note duration in seconds
        """
        self.quantize_resolution = quantize_resolution
        self.default_tempo = default_tempo
        self.min_note_duration = min_note_duration

        logger.info(
            f"MIDIProcessor initialized: "
            f"resolution={quantize_resolution}, tempo={default_tempo}"
        )

    def hz_to_midi(self, hz: float) -> int:
        """
        Convert frequency in Hz to MIDI note number

        Args:
            hz: Frequency in Hz

        Returns:
            MIDI note number (0-127)
        """
        if hz <= 0:
            return 0

        midi_note = int(round(69 + 12 * np.log2(hz / 440.0)))

        # Clamp to valid MIDI range
        return max(0, min(127, midi_note))

    def midi_to_hz(self, midi_note: int) -> float:
        """
        Convert MIDI note number to frequency

        Args:
            midi_note: MIDI note number

        Returns:
            Frequency in Hz
        """
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

    def quantize_time(
        self,
        time: float,
        tempo: float,
        resolution: int
    ) -> float:
        """
        Quantize time to rhythmic grid

        Args:
            time: Time in seconds
            tempo: Tempo in BPM
            resolution: Grid resolution (4, 8, 16, etc.)

        Returns:
            Quantized time in seconds
        """
        beats_per_second = tempo / 60.0
        total_beats = time * beats_per_second
        grid_divisions = total_beats * resolution
        quantized_divisions = round(grid_divisions)
        quantized_beats = quantized_divisions / resolution
        return quantized_beats / beats_per_second

    def detect_note_boundaries(
        self,
        pitches: List[float],
        confidences: List[float],
        times: List[float]
    ) -> List[Dict]:
        """
        Detect note boundaries from continuous pitch stream

        Args:
            pitches: List of pitch values (Hz)
            confidences: List of confidence values
            times: List of time stamps

        Returns:
            List of note dicts with start, end, pitch, velocity
        """
        notes = []

        # Filter out unvoiced frames
        voiced_indices = [i for i, (p, c) in enumerate(zip(pitches, confidences)) if p > 0 and c > 0.5]

        if not voiced_indices:
            return notes

        # Group consecutive similar pitches into notes
        current_note_start = voiced_indices[0]
        current_pitch = pitches[voiced_indices[0]]
        current_midi = self.hz_to_midi(current_pitch)
        note_pitches = [current_pitch]
        note_confidences = [confidences[voiced_indices[0]]]

        for i in range(1, len(voiced_indices)):
            idx = voiced_indices[i]
            pitch = pitches[idx]
            midi_note = self.hz_to_midi(pitch)
            confidence = confidences[idx]

            # Check if this is a new note (pitch changed significantly)
            if abs(midi_note - current_midi) > 1:  # More than 1 semitone change
                # Finalize current note
                start_time = times[current_note_start]
                end_time = times[voiced_indices[i - 1]]
                duration = end_time - start_time

                if duration >= self.min_note_duration:
                    notes.append({
                        'start': start_time,
                        'end': end_time,
                        'pitch': current_midi,
                        'velocity': self._calculate_velocity(note_confidences),
                        'duration': duration
                    })

                # Start new note
                current_note_start = idx
                current_pitch = pitch
                current_midi = midi_note
                note_pitches = [pitch]
                note_confidences = [confidence]
            else:
                # Continue current note
                note_pitches.append(pitch)
                note_confidences.append(confidence)

        # Don't forget the last note
        if note_pitches:
            start_time = times[current_note_start]
            end_time = times[voiced_indices[-1]]
            duration = end_time - start_time

            if duration >= self.min_note_duration:
                notes.append({
                    'start': start_time,
                    'end': end_time,
                    'pitch': current_midi,
                    'velocity': self._calculate_velocity(note_confidences),
                    'duration': duration
                })

        return notes

    def _calculate_velocity(self, confidences: List[float]) -> int:
        """
        Calculate MIDI velocity from confidence values

        Args:
            confidences: List of confidence values

        Returns:
            Velocity value (0-127)
        """
        if not confidences:
            return 100

        avg_confidence = np.mean(confidences)
        velocity = int(60 + avg_confidence * 60)  # Map 0-1 to 60-120
        return max(40, min(127, velocity))

    def detect_key_scale(
        self,
        midi: pretty_midi.PrettyMIDI
    ) -> Dict:
        """
        Detect key and scale using Krumhansl-Schmuckler algorithm

        Args:
            midi: PrettyMIDI object

        Returns:
            Dict with detected key, scale, and confidence
        """
        # Build pitch class histogram
        pitch_classes = np.zeros(12)

        for instrument in midi.instruments:
            for note in instrument.notes:
                pitch_class = note.pitch % 12
                pitch_classes[pitch_class] += note.velocity

        # Normalize
        if np.sum(pitch_classes) > 0:
            pitch_classes = pitch_classes / np.sum(pitch_classes)

        # Test against major and minor profiles
        best_correlation = -1
        best_key = None
        best_mode = None

        for root in range(12):
            # Shift profile to this root
            major_shifted = np.roll(self.MAJOR_PROFILE, root)
            minor_shifted = np.roll(self.MINOR_PROFILE, root)

            # Normalize profiles
            major_shifted = major_shifted / np.sum(major_shifted)
            minor_shifted = minor_shifted / np.sum(minor_shifted)

            # Calculate correlations
            major_corr = np.corrcoef(pitch_classes, major_shifted)[0, 1]
            minor_corr = np.corrcoef(pitch_classes, minor_shifted)[0, 1]

            if major_corr > best_correlation:
                best_correlation = major_corr
                best_key = root
                best_mode = 'major'

            if minor_corr > best_correlation:
                best_correlation = minor_corr
                best_key = root
                best_mode = 'minor'

        return {
            'key': self.NOTE_NAMES[best_key] if best_key is not None else 'C',
            'scale': best_mode or 'major',
            'confidence': float(max(0, best_correlation)) if best_correlation is not None else 0.0
        }

    def pitches_to_midi(
        self,
        pitch_data: Dict,
        output_path: str,
        quantize: bool = True
    ) -> Dict:
        """
        Convert pitch data to MIDI file

        Args:
            pitch_data: Dictionary from pitch detector with times, pitches, confidences
            output_path: Path to save MIDI file
            quantize: Whether to quantize timing

        Returns:
            Dict with MIDI metadata
        """
        logger.info(f"Converting pitch data to MIDI: {output_path}")

        times = pitch_data.get('times', [])
        pitches = pitch_data.get('pitches', [])
        confidences = pitch_data.get('confidences', [])

        if not times or not pitches:
            raise ValueError("Empty pitch data provided")

        # Detect note boundaries
        notes = self.detect_note_boundaries(pitches, confidences, times)

        if not notes:
            raise ValueError("No valid notes detected from pitch data")

        # Create PrettyMIDI object
        midi = pretty_midi.PrettyMIDI(initial_tempo=self.default_tempo)

        # Create an instrument (Acoustic Grand Piano)
        instrument = pretty_midi.Instrument(
            program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'),
            is_drum=False
        )

        # Add notes to instrument
        for note_data in notes:
            start_time = note_data['start']
            end_time = note_data['end']

            # Quantize if requested
            if quantize:
                start_time = self.quantize_time(
                    start_time,
                    self.default_tempo,
                    self.quantize_resolution
                )
                end_time = self.quantize_time(
                    end_time,
                    self.default_tempo,
                    self.quantize_resolution
                )

            # Ensure minimum duration
            if end_time - start_time < self.min_note_duration:
                end_time = start_time + self.min_note_duration

            note = pretty_midi.Note(
                velocity=note_data['velocity'],
                pitch=note_data['pitch'],
                start=start_time,
                end=end_time
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)

        # Detect key and scale
        key_info = self.detect_key_scale(midi)

        # Save MIDI file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        midi.write(output_path)

        result = {
            "midi_file": output_path,
            "detected_key": key_info['key'],
            "scale": key_info['scale'],
            "confidence": key_info['confidence'],
            "note_count": len(notes),
            "duration": midi.get_end_time(),
            "tempo": self.default_tempo,
            "notes": [
                {
                    "pitch": n['pitch'],
                    "start": round(n['start'], 3),
                    "end": round(n['end'], 3),
                    "velocity": n['velocity']
                }
                for n in notes
            ]
        }

        logger.info(
            f"MIDI saved: {output_path} "
            f"({len(notes)} notes, key={key_info['key']} {key_info['scale']})"
        )

        return result

    def get_note_sequence(self, midi_path: str) -> List[Dict]:
        """
        Extract note sequence from MIDI file

        Args:
            midi_path: Path to MIDI file

        Returns:
            List of note dicts
        """
        midi = pretty_midi.PrettyMIDI(midi_path)
        notes = []

        for instrument in midi.instruments:
            for note in instrument.notes:
                notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'end': note.end,
                    'velocity': note.velocity,
                    'duration': note.end - note.start
                })

        # Sort by start time
        notes.sort(key=lambda x: x['start'])
        return notes

    def transpose_to_key(
        self,
        midi_path: str,
        target_key: str,
        output_path: str
    ) -> pretty_midi.PrettyMIDI:
        """
        Transpose MIDI to target key

        Args:
            midi_path: Input MIDI path
            target_key: Target key (e.g., 'C', 'D', 'F#')
            output_path: Output MIDI path

        Returns:
            Transposed PrettyMIDI object
        """
        midi = pretty_midi.PrettyMIDI(midi_path)

        # Get current key
        key_info = self.detect_key_scale(midi)
        current_key = key_info['key']

        # Calculate semitone difference
        current_idx = self.NOTE_NAMES.index(current_key)
        target_idx = self.NOTE_NAMES.index(target_key)
        semitones = target_idx - current_idx

        # Transpose all notes
        for instrument in midi.instruments:
            for note in instrument.notes:
                note.pitch = max(0, min(127, note.pitch + semitones))

        # Save
        midi.write(output_path)

        logger.info(f"Transposed MIDI from {current_key} to {target_key} ({semitones} semitones)")

        return midi


# Standalone test
def test_midi_processor():
    """Test MIDI processor with sample data"""
    processor = MIDIProcessor()

    # Create sample pitch data (C major scale)
    base_freq = 261.63  # C4
    times = list(range(100))
    pitches = []
    confidences = []

    for i, time in enumerate(times):
        # Play each note for about 10 frames
        note_idx = i // 10
        if note_idx < 8:
            # C major scale: C, D, E, F, G, A, B, C
            semitones = [0, 2, 4, 5, 7, 9, 11, 12][note_idx]
            freq = base_freq * (2 ** (semitones / 12))
            pitches.append(freq)
            confidences.append(0.9)
        else:
            pitches.append(0)
            confidences.append(0)

    pitch_data = {
        'times': [t * 0.01 for t in times],
        'pitches': pitches,
        'confidences': confidences
    }

    # Convert to MIDI
    result = processor.pitches_to_midi(pitch_data, 'test_output.mid')

    print(f"Created MIDI: {result}")
    print(f"Key: {result['detected_key']} {result['scale']}")
    print(f"Notes: {result['note_count']}")

    # Cleanup
    Path('test_output.mid').unlink(missing_ok=True)


if __name__ == "__main__":
    test_midi_processor()
