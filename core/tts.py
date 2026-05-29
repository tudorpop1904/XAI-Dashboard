"""
tts.py — Browser-native Text-to-Speech via Web Speech API.

Injects JavaScript into the Streamlit page to trigger speech synthesis.
Zero server-side dependencies — works entirely in the browser.
"""

import streamlit.components.v1 as components


def speak(
    text: str,
    rate: float = 0.85,
    pitch: float = 1.0,
    lang: str = "en-US",
    auto_play: bool = True,
) -> None:
    """
    Speak text using the browser's Web Speech API.

    Parameters
    ----------
    text : str      Text to speak.
    rate : float    Speech rate (0.5 = slow, 1.0 = normal, 2.0 = fast).
    pitch : float   Pitch (0.0 = low, 1.0 = normal, 2.0 = high).
    lang : str      BCP 47 language tag.
    auto_play : bool  If True, starts speaking immediately on render.
    """
    # Escape text for JS string literal
    safe_text = text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")

    js = f"""
    <script>
    (function() {{
        if ('speechSynthesis' in window) {{
            // Cancel any ongoing speech first
            window.speechSynthesis.cancel();

            var utterance = new SpeechSynthesisUtterance('{safe_text}');
            utterance.rate = {rate};
            utterance.pitch = {pitch};
            utterance.lang = '{lang}';

            {"window.speechSynthesis.speak(utterance);" if auto_play else ""}
        }}
    }})();
    </script>
    """
    components.html(js, height=0, width=0)


def speak_character_result(
    character: str,
    confidence: float,
    is_correct: bool | None = None,
) -> None:
    """
    Announce a character recognition result via TTS.

    Designed for visually impaired children — clear, encouraging, slow pace.
    """
    conf_pct = int(confidence * 100)

    if is_correct is True:
        msg = f"Great job! You wrote the letter {character}. I'm {conf_pct} percent sure. Well done!"
    elif is_correct is False:
        msg = f"I see the letter {character}, but that doesn't seem right. Let's try again!"
    else:
        msg = f"I see the letter {character}. I'm {conf_pct} percent confident."

    speak(msg, rate=0.75, pitch=1.1)


def speak_exercise(exercise_text: str) -> None:
    """Read out a writing exercise prompt slowly and clearly."""
    speak(f"Here is your next exercise. {exercise_text}", rate=0.7, pitch=1.0)
