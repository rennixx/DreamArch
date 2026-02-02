"""
Test the FREE pipeline - Lyrics (Ollama) + Vocals (pyttsx3)
No API keys required!
"""

import os
import sys

print("=== Dream Architect - FREE Pipeline Test ===")
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

    print("✅ Lyrics generated:")
    for line in result['lyrics']:
        print(f"   {line}")

except Exception as e:
    print(f"❌ Lyrics failed: {e}")
    sys.exit(1)

print()

# Test Vocal Synthesis with pyttsx3
print("[2/2] Testing Vocals with pyttsx3...")
try:
    from modules.vocal_synth import VocalSynthesizer

    synth = VocalSynthesizer(provider='pyttsx3')

    # Simple test
    test_lyrics = ["Testing the vocal synthesis", "Everything is working great"]
    test_path = "test_vocals_py_ttsx3.wav"

    synth.synthesize_vocals(
        lyrics=test_lyrics,
        melody_midi="",
        style="Pink Floyd",
        output_path=test_path
    )

    print(f"✅ Vocals saved to: {test_path}")

    # Cleanup
    import os
    if os.path.exists(test_path):
        os.unlink(test_path)
        print("   (test file cleaned up)")

except Exception as e:
    print(f"❌ Vocals failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 50)
print("🎉 FREE PIPELINE WORKING!")
print("=" * 50)
print()
print("Your setup is ready for:")
print("  ✅ Lyrics: Ollama (ministral-3)")
print("  ✅ Vocals: pyttsx3 (free)")
print()
print("To add MusicGen:")
print("  1. Install Visual Studio Build Tools:")
print("     https://visualstudio.microsoft.com/visual-cpp-build-tools/")
print("  2. Install 'Desktop development with C++'")
print("  3. Run: pip install audiocraft")
