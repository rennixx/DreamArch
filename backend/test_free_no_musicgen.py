"""
Test the FREE pipeline without MusicGen
- Lyrics (Ollama)
- Vocals (pyttsx3)
- Skips MusicGen instrumental generation
"""

import os
import sys

print("=== Dream Architect - FREE Pipeline Test (No MusicGen) ===")
print()

# Test Lyrics Generation with Ollama
print("[1/2] Testing Lyrics with Ollama...")
try:
    from modules.lyrics_generator import LyricsGenerator

    gen = LyricsGenerator(
        provider='ollama',
        model='ministral-3:latest'
    )

    result = gen.generate_lyrics(
        style='Pink Floyd',
        mood='contemplative',
        theme='space',
        num_lines=4
    )

    print("[OK] Lyrics generated:")
    for line in result['lyrics']:
        print(f"   {line}")

except Exception as e:
    print(f"[FAIL] Lyrics failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test Vocal Synthesis with pyttsx3
print("[2/2] Testing Vocals with pyttsx3...")
try:
    from modules.vocal_synth import VocalSynthesizer

    synth = VocalSynthesizer(provider='pyttsx3')

    # Use the generated lyrics
    test_lyrics = result['lyrics'][:2]  # First 2 lines for quick test
    test_path = "test_vocals_free.wav"

    synth.synthesize_vocals(
        lyrics=test_lyrics,
        melody_midi="",
        style="Pink Floyd",
        output_path=test_path
    )

    print(f"[OK] Vocals saved to: {test_path}")

    # Cleanup
    if os.path.exists(test_path):
        os.unlink(test_path)
        print("   (test file cleaned up)")

except Exception as e:
    print(f"[FAIL] Vocals failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 50)
print("FREE PIPELINE WORKING!")
print("=" * 50)
print()
print("[OK] Lyrics: Ollama (ministral-3)")
print("[OK] Vocals: pyttsx3 (free)")
print("[PAUSED] MusicGen: Requires C++ build tools")
print()
print("To enable MusicGen instrumental generation:")
print("  1. Install Visual Studio Build Tools:")
print("     https://visualstudio.microsoft.com/visual-cpp-build-tools/")
print("  2. Install 'Desktop development with C++'")
print("  3. Reinstall audiocraft with proper dependencies")
