
"""Text-to-speech helper using pyttsx3."""
import os
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

def synthesize_tts(text: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if pyttsx3 is not None:
        engine = pyttsx3.init()
        rate = engine.getProperty('rate')
        engine.setProperty('rate', int(rate * 0.95))
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        return out_path
    else:
        raise RuntimeError("pyttsx3 not installed; please install pyttsx3 or provide alternate TTS.")
