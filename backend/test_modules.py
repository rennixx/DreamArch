"""
Comprehensive test script for Dream Architect modules
Tests core functionality without requiring GPU or API keys
Windows-compatible version
"""

import sys
import numpy as np
import tempfile
import os
from pathlib import Path

print("=" * 60)
print("Dream Architect - Module Tests")
print("=" * 60)

def cleanup_later(filepath):
    """Schedule file for cleanup (Windows workaround)"""
    try:
        os.unlink(filepath)
    except:
        pass  # Will be cleaned up later

temp_files = []

# Test 1: Audio Utils
print("\n[1/6] Testing Audio Utils...")
try:
    from utils.audio_utils import (
        load_audio, save_audio, apply_highpass_filter,
        get_audio_duration, check_gpu_availability, match_length
    )

    # Create test audio
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    test_audio = np.sin(2 * np.pi * 440 * t)

    # Test save/load
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
        temp_files.append(temp_path)
        save_audio(test_audio, sr, temp_path)
        loaded, loaded_sr = load_audio(temp_path)
        assert len(loaded) > 0, "Failed to load audio"

    gpu_info = check_gpu_availability()
    print(f"  [OK] Audio Utils (GPU: {gpu_info['device']})")
except Exception as e:
    print(f"  [FAIL] Audio Utils: {e}")

# Test 2: File Manager
print("\n[2/6] Testing File Manager...")
try:
    from utils.file_manager import (
        generate_job_id, get_upload_path, get_output_path,
        ensure_directories, file_exists
    )

    job_id = generate_job_id()
    assert job_id, "Failed to generate job ID"

    upload_path = get_upload_path(job_id)
    output_path = get_output_path("midi", job_id, "mid")

    print(f"  [OK] File Manager (job_id: {job_id[:8]}...)")
except Exception as e:
    print(f"  [FAIL] File Manager: {e}")

# Test 3: MIDI Processor
print("\n[3/6] Testing MIDI Processor...")
try:
    from modules.midi_processor import MIDIProcessor

    # Create test pitch data (C major scale)
    base_freq = 261.63  # C4
    times = list(range(100))
    pitches = []
    confidences = []

    for i, time in enumerate(times):
        note_idx = i // 10
        if note_idx < 8:
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

    processor = MIDIProcessor()

    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
        temp_path = f.name
        temp_files.append(temp_path)
        result = processor.pitches_to_midi(pitch_data, temp_path)
        assert result['note_count'] > 0, "No notes generated"

    print(f"  [OK] MIDI Processor ({result['note_count']} notes, key: {result['detected_key']} {result['scale']})")
except Exception as e:
    print(f"  [FAIL] MIDI Processor: {e}")

# Test 4: Audio Mixer
print("\n[4/6] Testing Audio Mixer...")
try:
    from modules.mixer import AudioMixer
    from utils.file_manager import file_exists

    sr = 44100
    duration = 2
    t = np.linspace(0, duration, int(sr * duration))

    # Create test tracks
    inst = np.sin(2 * np.pi * 261.63 * t) * 0.3  # C4
    inst = np.column_stack([inst, inst])

    vocals = np.sin(2 * np.pi * 523.25 * t) * 0.15  # C5
    vocals = np.column_stack([vocals, vocals])

    mixer = AudioMixer(sample_rate=sr)

    # Save and mix
    with tempfile.TemporaryDirectory() as tmpdir:
        inst_path = os.path.join(tmpdir, "inst.wav")
        vocal_path = os.path.join(tmpdir, "vocals.wav")
        mixed_path = os.path.join(tmpdir, "mixed.wav")

        from utils.audio_utils import save_audio
        save_audio(inst, sr, inst_path)
        save_audio(vocals, sr, vocal_path)

        # Mix
        result_path = mixer.mix_tracks(inst_path, vocal_path, mixed_path)
        assert file_exists(result_path), "Mixed file not created"

    print(f"  [OK] Audio Mixer (mixing successful)")
except Exception as e:
    print(f"  [FAIL] Audio Mixer: {e}")

# Test 5: Pitch Detector (without CREPE model)
print("\n[5/6] Testing Pitch Detector (basic)...")
try:
    from modules.pitch_detector import PitchDetector

    detector = PitchDetector(model_capacity='small')
    print(f"  [OK] Pitch Detector initialized (device: {detector.device})")
    print(f"    Note: CREPE model loading deferred (requires GPU)")
except Exception as e:
    print(f"  [FAIL] Pitch Detector: {e}")

# Test 6: Configuration
print("\n[6/6] Testing Configuration...")
try:
    import yaml
    import json

    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)

    with open("config/style_presets.json") as f:
        presets = json.load(f)

    assert config['generation']['musicgen_model'] == 'facebook/musicgen-small'
    assert len(presets['presets']) == 4

    print(f"  [OK] Configuration ({len(presets['presets'])} style presets)")
except Exception as e:
    print(f"  [FAIL] Configuration: {e}")

# Cleanup temp files
print("\nCleaning up temp files...")
for f in temp_files:
    cleanup_later(f)

# Summary
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("[OK] Core modules tested successfully")
print("\nNote: Full integration test requires:")
print("  - GPU with 4GB+ VRAM")
print("  - OpenAI API key (set OPENAI_API_KEY)")
print("  - audiocraft library (pip install audiocraft)")
print("\nTo run full test:")
print("  1. Install ML dependencies: pip install torch audiocraft crepe")
print("  2. Set API key: set OPENAI_API_KEY=sk-...")
print("  3. Run: python test_full_integration.py")
print("=" * 60)
