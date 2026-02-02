"""
Demo Generation Script
Creates demo tracks for showcasing Dream Architect capabilities
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are available"""
    missing = []

    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, generation will be slow")
    except ImportError:
        missing.append("torch")

    try:
        import audiocraft
    except ImportError:
        missing.append("audiocraft")

    try:
        import openai
    except ImportError:
        missing.append("openai")

    return missing


def create_demo_track(
    style: str = "pink-floyd",
    melody_duration: int = 3,
    output_duration: int = 30
):
    """
    Create a demo track

    Args:
        style: Style preset ID
        melody_duration: Duration of input melody (seconds)
        output_duration: Duration of generated track (seconds)
    """
    from modules.pitch_detector import PitchDetector
    from modules.midi_processor import MIDIProcessor
    from modules.music_generator import MusicGenerator
    from modules.lyrics_generator import LyricsGenerator
    from modules.vocal_synth import VocalSynthesizer
    from modules.mixer import AudioMixer
    from utils.file_manager import generate_job_id, get_output_path
    from utils.audio_analyzer import AudioAnalyzer, benchmark_generation_time
    import yaml
    import json

    print("\n" + "=" * 60)
    print("DREAM ARCHITECT - DEMO GENERATION")
    print("=" * 60)

    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"\n[ERROR] Missing dependencies: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return None

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[ERROR] OPENAI_API_KEY not set")
        print("Set with: set OPENAI_API_KEY=sk-...")
        return None

    # Load config
    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)

    with open("config/style_presets.json") as f:
        presets = json.load(f)["presets"]

    preset = next((p for p in presets if p["id"] == style), presets[0])

    print(f"\nStyle: {preset['name']}")
    print(f"Description: {preset['description']}")

    job_id = generate_job_id()
    print(f"Job ID: {job_id}")

    print("\n--- Pipeline Started ---")

    # Step 1: Create test melody
    print("\n[1/7] Creating test melody...")
    import numpy as np
    import soundfile as sf

    sr = 44100
    duration = melody_duration
    t = np.linspace(0, duration, int(sr * duration))

    # Simple memorable melody
    melody_freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
    melody_audio = np.zeros_like(t)

    notes_per_second = len(melody_freqs) / duration
    for i, freq in enumerate(melody_freqs):
        start_sample = int(i * (sr / notes_per_second))
        end_sample = int((i + 1) * (sr / notes_per_second))
        note_t = t[start_sample:end_sample] - t[start_sample]
        envelope = np.exp(-3 * note_t)  # Decay envelope
        melody_audio[start_sample:end_sample] += np.sin(2 * np.pi * freq * note_t) * envelope * 0.5

    upload_path = get_upload_path(job_id, "wav")
    sf.write(upload_path, melody_audio, sr)
    print(f"  Created {duration}s melody")

    # Step 2: Extract pitch
    print("\n[2/7] Extracting pitch...")
    pitch_detector = PitchDetector(model_capacity='small')

    try:
        pitch_data, pitch_time = benchmark_generation_time(
            pitch_detector.extract_pitches,
            upload_path
        )
        print(f"  Extracted in {pitch_time:.2f}s")
    except Exception as e:
        print(f"  Skipped (CREPE not available): {e}")
        # Create dummy pitch data
        pitch_data = {
            'times': t[:len(t)//10].tolist(),
            'pitches': melody_freqs * (len(t) // len(melody_freqs) + 1),
            'confidences': [0.9] * len(t)
        }

    # Step 3: Convert to MIDI
    print("\n[3/7] Converting to MIDI...")
    midi_processor = MIDIProcessor()
    midi_path = get_output_path("midi", job_id, "mid")
    midi_data = midi_processor.pitches_to_midi(pitch_data, midi_path)
    print(f"  Created MIDI with {midi_data['note_count']} notes")

    # Step 4: Generate instrumental
    print("\n[4/7] Generating instrumental...")
    music_gen = MusicGenerator(
        model_name=config['generation']['musicgen_model'],
        use_fp16=config['generation']['use_fp16']
    )

    try:
        instrumental_audio, sr = benchmark_generation_time(
            music_gen.generate_from_melody,
            midi_path=midi_path,
            style_prompt=preset['prompt'],
            duration=min(output_duration, 30)
        )
        instrumental_path = get_output_path("instrumental", job_id, "wav")
        music_gen.save_generated_audio(instrumental_audio, sr, instrumental_path)
        print(f"  Generated {len(instrumental_audio)/sr:.1f}s instrumental")
    except Exception as e:
        print(f"  Failed: {e}")
        return None

    # Clear GPU cache
    music_gen.clear_cache()

    # Step 5: Generate lyrics
    print("\n[5/7] Generating lyrics...")
    lyrics_gen = LyricsGenerator(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        lyrics_data, lyrics_time = benchmark_generation_time(
            lyrics_gen.generate_lyrics,
            preset['name'],
            "contemplative",
            None,
            6
        )
        print(f"  Generated {len(lyrics_data['lyrics'])} lines in {lyrics_time:.2f}s")
        for line in lyrics_data['lyrics']:
            print(f"    {line}")
    except Exception as e:
        print(f"  Failed: {e}")
        # Use fallback lyrics
        lyrics_data = {'lyrics': ["Dreams of electric sheep", "Dancing in the deep", "Lost in cosmic space", "Time we cannot keep"]}

    # Step 6: Generate vocals
    print("\n[6/7] Generating vocals...")
    vocal_synth = VocalSynthesizer(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        vocals_path = get_output_path("vocals", job_id, "wav")
        vocal_synth.synthesize_vocals(
            lyrics_data['lyrics'],
            midi_path,
            preset['name'],
            vocals_path
        )

        vocals_processed_path = get_output_path("vocals", f"{job_id}_processed", "wav")
        vocal_synth.apply_vocal_effects(
            vocals_path,
            preset['name'],
            vocals_processed_path
        )
        print("  Vocals generated and processed")
    except Exception as e:
        print(f"  Failed: {e}")
        vocals_processed_path = None

    # Step 7: Mix and master
    print("\n[7/7] Mixing and mastering...")
    mixer = AudioMixer(sample_rate=sr)

    mixed_path = get_output_path("mixed", job_id, "wav")
    mixer.mix_tracks(
        instrumental_path=instrumental_path,
        vocal_path=vocals_processed_path,
        output_path=mixed_path
    )

    final_path = get_output_path("final", job_id, "mp3")
    mixer.master_track(
        mixed_path,
        final_path,
        target_loudness=config['mixing']['target_loudness']
    )

    print(f"  Final track: {final_path}")

    # Analyze result
    print("\n--- Quality Analysis ---")
    analyzer = AudioAnalyzer()
    try:
        analysis = analyzer.analyze_file(final_path)
        analyzer.print_report(analysis)
    except:
        print("  Analysis skipped")

    print("\n" + "=" * 60)
    print("DEMO GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {final_path}")
    print(f"Style: {preset['name']}")
    print(f"Duration: {output_duration}s")
    print("\nTo play:")
    print(f"  Start: {final_path}")

    return final_path


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Dream Architect demo")
    parser.add_argument(
        "--style",
        choices=["pink-floyd", "daft-punk", "billie-eilish", "frank-zappa"],
        default="pink-floyd",
        help="Style preset"
    )
    parser.add_argument(
        "--melody-duration",
        type=int,
        default=3,
        help="Input melody duration (seconds)"
    )
    parser.add_argument(
        "--output-duration",
        type=int,
        default=30,
        help="Output track duration (seconds)"
    )

    args = parser.parse_args()

    # Change to backend directory
    os.chdir(Path(__file__).parent)

    create_demo_track(
        style=args.style,
        melody_duration=args.melody_duration,
        output_duration=args.output_duration
    )


if __name__ == "__main__":
    main()
