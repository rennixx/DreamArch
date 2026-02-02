"""
Lyrics Generator Module using OpenAI GPT-3.5-turbo
Fast lyrics generation with style-specific prompting

Optimized for speed and cost-efficiency.
"""

import os
from typing import Dict, List, Optional
import logging
from pathlib import Path
import json

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LyricsGenerator:
    """
    Generate lyrics using OpenAI's GPT models

    Features:
    - Style-specific prompts
    - Syllable counting
    - Rhyme detection
    - Result caching
    - Fast GPT-3.5-turbo for MVP
    """

    # Style-specific characteristics for prompt engineering
    STYLE_CHARACTERISTICS = {
        "Pink Floyd": """
            STYLE: Pink Floyd
            - Philosophical, existential themes
            - Surreal, poetic imagery
            - Often about time, alienation, madness, cosmic journeys
            - Abstract metaphors
            - Contemplative, sometimes dark
            - Example: "Breathe, breathe in the air / Don't be afraid to care"
        """,
        "Billie Eilish": """
            STYLE: Billie Eilish
            - Intimate, confessional tone
            - Dark pop sensibility
            - First-person perspective
            - Modern, conversational language
            - Themes of anxiety, relationships, identity
            - Example: "I'm the bad guy, duh"
        """,
        "Daft Punk": """
            STYLE: Daft Punk
            - Repetitive, hook-focused
            - Robotic, filtered vocals aesthetic
            - Simple, catchy phrases
            - Themes of technology, nightlife, movement, celebration
            - Example: "One more time / We're gonna celebrate"
        """,
        "Frank Zappa": """
            STYLE: Frank Zappa
            - Satirical, absurd humor
            - Complex wordplay
            - Social commentary
            - Irreverent, quirky
            - Non-sequiturs and unexpected twists
            - Example: "Don't eat the yellow snow"
        """
    }

    # System prompt for the lyricist persona
    SYSTEM_PROMPT = """You are a professional lyricist specializing in diverse musical styles.
You write authentic, evocative lyrics that capture the essence of each genre.
You understand syllable counts, rhyme schemes, and poetic devices.
Your lyrics are original and should not copy existing songs.
Write ONLY the lyrics, no titles, explanations, or metadata."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        cache_enabled: bool = True
    ):
        """
        Initialize lyrics generator

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use (gpt-3.5-turbo for speed, gpt-4 for quality)
            cache_enabled: Enable result caching
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package is required. Install with: pip install openai"
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.cache_enabled = cache_enabled
        self.cache = {}

        logger.info(f"LyricsGenerator initialized with model={model}")

    def _build_prompt(
        self,
        style: str,
        mood: str,
        theme: Optional[str],
        num_lines: int
    ) -> str:
        """
        Build prompt for GPT

        Args:
            style: Musical style/artist reference
            mood: Desired mood (contemplative, energetic, etc.)
            theme: Optional theme for lyrics
            num_lines: Number of lines to generate

        Returns:
            Formatted prompt string
        """
        style_info = self.STYLE_CHARACTERISTICS.get(style, "")
        theme_text = f"Theme: {theme}" if theme else "Theme: open-ended, abstract"

        prompt = f"""Write {num_lines} lines of lyrics in the style of {style}.

SPECIFICATIONS:
- Mood: {mood}
- {theme_text}
- Line count: {num_lines} lines exactly
- Structure: verse-like (not chorus)
- Syllables per line: 8-12 (singable)
- Original content only

{style_info}

REQUIREMENTS:
- Write ONLY the lyrics, nothing else
- Each line on a new line
- No title, no explanations, no metadata
- Make it singable and rhythmic
- Use appropriate rhyme scheme (AABB, ABAB, or ABCB)"""

        return prompt

    def generate_lyrics(
        self,
        style: str,
        mood: str = "contemplative",
        theme: Optional[str] = None,
        num_lines: int = 6,
        temperature: float = 0.8
    ) -> Dict:
        """
        Generate lyrics for a given style and mood

        Args:
            style: Musical style (Pink Floyd, Billie Eilish, etc.)
            mood: Mood of the lyrics
            theme: Optional theme to focus on
            num_lines: Number of lines to generate
            temperature: Sampling temperature (lower = more focused)

        Returns:
            Dict with lyrics, syllable counts, rhyme scheme, metadata
        """
        # Check cache
        cache_key = f"{style}_{mood}_{theme}_{num_lines}_{temperature}"
        if self.cache_enabled and cache_key in self.cache:
            logger.info(f"Using cached lyrics for: {style}")
            return self.cache[cache_key]

        logger.info(f"Generating lyrics: style={style}, mood={mood}, lines={num_lines}")

        # Build prompt
        prompt = self._build_prompt(style, mood, theme, num_lines)

        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=500,
                frequency_penalty=0.5,  # Reduce repetition
                presence_penalty=0.3    # Encourage variety
            )

            # Extract lyrics
            lyrics_text = response.choices[0].message.content.strip()

            # Parse into lines
            lyrics = self._parse_lyrics(lyrics_text, num_lines)

            # Analyze lyrics
            syllable_counts = [self.count_syllables(line) for line in lyrics]
            total_syllables = sum(syllable_counts)
            rhyme_scheme = self._detect_rhyme_scheme(lyrics)

            result = {
                "lyrics": lyrics,
                "syllable_count": syllable_counts,
                "total_syllables": total_syllables,
                "rhyme_scheme": rhyme_scheme,
                "style": style,
                "mood": mood,
                "theme": theme,
                "model": self.model,
                "estimated_duration": total_syllables / 4  # Rough estimate at 4 syllables/sec
            }

            # Cache result
            if self.cache_enabled:
                self.cache[cache_key] = result

            logger.info(
                f"Generated {len(lyrics)} lines "
                f"({total_syllables} syllables, rhyme: {rhyme_scheme})"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to generate lyrics: {e}")
            raise

    def _parse_lyrics(self, text: str, expected_lines: int) -> List[str]:
        """
        Parse generated text into clean lyric lines

        Args:
            text: Raw text from API
            expected_lines: Expected number of lines

        Returns:
            List of cleaned lyric lines
        """
        # Split by lines
        lines = text.strip().split('\n')

        # Clean up
        cleaned = []
        for line in lines:
            line = line.strip()
            # Skip empty lines and metadata
            if line and not line.startswith(('Here are', 'Lyrics:', 'Title:', 'Verse:')):
                # Remove leading dashes/numbers
                line = line.lstrip('-*0123456789.) ')
                if line:
                    cleaned.append(line)

        # Ensure we have the right number of lines
        if len(cleaned) > expected_lines:
            cleaned = cleaned[:expected_lines]
        elif len(cleaned) < expected_lines:
            # Pad with empty lines if needed
            cleaned.extend([''] * (expected_lines - len(cleaned)))

        return cleaned

    def count_syllables(self, text: str) -> int:
        """
        Count syllables in text using a simple algorithm

        Args:
            text: Text to count syllables in

        Returns:
            Approximate syllable count
        """
        if not text:
            return 0

        # Remove non-alphabetic characters
        text = ''.join(c for c in text if c.isalpha() or c.isspace())

        words = text.lower().split()
        count = 0

        for word in words:
            # Count vowel groups
            vowels = 'aeiouy'
            prev_was_vowel = False

            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    count += 1
                prev_was_vowel = is_vowel

            # Handle silent 'e' at end
            if word.endswith('e') and count > 1:
                count -= 1

        return max(1, count)

    def _detect_rhyme_scheme(self, lines: List[str]) -> str:
        """
        Detect rhyme scheme from lyrics

        Args:
            lines: List of lyric lines

        Returns:
            Rhyme scheme as string (e.g., "AABB", "ABAB")
        """
        if len(lines) < 2:
            return ""

        # Get ending sounds for each line
        endings = [self._get_rhyme_sound(line) for line in lines]

        # Assign letters to rhymes
        scheme = []
        rhyme_map = {}
        current_letter = ord('A')

        for i, ending in enumerate(endings):
            if not ending:  # Empty line or no sound
                scheme.append('X')
                continue

            if ending not in rhyme_map:
                rhyme_map[ending] = chr(current_letter)
                current_letter += 1

            scheme.append(rhyme_map[ending])

        return ''.join(scheme)

    def _get_rhyme_sound(self, line: str) -> str:
        """
        Extract the rhyming part (ending sound) from a line

        Args:
            line: Lyric line

        Returns:
            Rhyme sound (last few phonemes)
        """
        if not line:
            return ""

        # Simple approach: last word's ending
        words = line.lower().split()
        if not words:
            return ""

        last_word = words[-1]
        # Remove punctuation
        last_word = ''.join(c for c in last_word if c.isalpha())

        if len(last_word) < 2:
            return last_word

        # Return last 2-3 characters as rhyme sound
        return last_word[-min(3, len(last_word)):]

    def adapt_lyrics_to_melody(
        self,
        lyrics: List[str],
        melody_notes: List[Dict],
        target_duration: float
    ) -> List[Dict]:
        """
        Time lyrics to fit melody duration

        Args:
            lyrics: List of lyric lines
            melody_notes: List of note dicts with pitch, start, end
            target_duration: Target total duration

        Returns:
            List of timed lyric segments
        """
        total_syllables = sum(self.count_syllables(line) for line in lyrics)
        seconds_per_syllable = target_duration / total_syllables if total_syllables > 0 else 0.5

        timed_lyrics = []
        current_time = 0.0

        for line in lyrics:
            syllables = self.count_syllables(line)
            duration = syllables * seconds_per_syllable

            timed_lyrics.append({
                "text": line,
                "start": current_time,
                "end": current_time + duration,
                "syllables": syllables
            })

            current_time += duration + 0.2  # Add small pause between lines

        return timed_lyrics

    def clear_cache(self):
        """Clear the lyrics cache"""
        self.cache.clear()
        logger.info("Lyrics cache cleared")


# Standalone test
def test_lyrics_generator():
    """Test lyrics generator"""
    import os

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping test: OPENAI_API_KEY not set")
        return

    generator = LyricsGenerator(model="gpt-3.5-turbo")

    # Test generation
    result = generator.generate_lyrics(
        style="Pink Floyd",
        mood="contemplative",
        theme="cosmic journey",
        num_lines=6
    )

    print("Generated Lyrics:")
    for i, line in enumerate(result['lyrics'], 1):
        print(f"  {i}. {line}")

    print(f"\nSyllables: {result['syllable_count']}")
    print(f"Rhyme scheme: {result['rhyme_scheme']}")
    print(f"Estimated duration: {result['estimated_duration']:.1f}s")


if __name__ == "__main__":
    test_lyrics_generator()
