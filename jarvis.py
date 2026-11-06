"""
Jarvis - Step 1: Voice loop (free, runs on your laptop)

Flow:  microphone -> Whisper (speech to text) -> Ollama (brain) -> voice reply

Run:   python jarvis.py
Stop:  say "exit" / "bye jarvis" / "band karo", or press Ctrl+C
"""

import asyncio
import collections
import os
import re
import sys
import tempfile
import time

import numpy as np
import requests
import sounddevice as sd
from faster_whisper import WhisperModel

# ----------------------------- SETTINGS ------------------------------------
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"      # small model, fits 8GB RAM
WHISPER_MODEL = "base"            # "tiny" = faster, "base" = more accurate
LANGUAGE = None                   # None = auto detect (Urdu / English). Or "ur" / "en"

# Voices (edge-tts: free, needs internet). If it fails, offline Windows voice is used.
URDU_VOICE = "ur-PK-UzmaNeural"       # girl voice, Urdu
ENGLISH_VOICE = "en-US-JennyNeural"   # girl voice, English

# Microphone settings
SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.03
SILENCE_SECONDS = 0.9      # stop recording after this much silence
MAX_SECONDS = 20           # longest single sentence
MIN_SECONDS = 0.4          # ignore tiny noises

EXIT_WORDS = ["exit", "quit", "bye jarvis", "goodbye jarvis", "band karo",
              "بند کرو", "الوداع"]

SYSTEM_PROMPT = (
    "You are Jarvis, a friendly personal voice assistant, like the one in Iron Man. "
    "The user talks to you by voice and you answer by voice, so keep replies SHORT "
    "(1 to 3 sentences), natural and spoken. No markdown, no lists, no emojis. "
    "If the user speaks English, reply in English. "
    "If the user speaks Urdu or Hindi, reply in Urdu script (اردو). "
    "Be warm, a little witty, and helpful."
)
# ---------------------------------------------------------------------------

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ------------------------------ EARS ---------------------------------------
def calibrate_noise(seconds=1.0):
    """Measure room noise so we know what counts as 'speech'."""
    block = int(SAMPLE_RATE * BLOCK_SECONDS)
    levels = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=block) as stream:
        for _ in range(int(seconds / BLOCK_SECONDS)):
            data, _ = stream.read(block)
            levels.append(float(np.sqrt(np.mean(data[:, 0] ** 2))))
    ambient = float(np.mean(levels))
    return max(0.012, ambient * 3.0)


def record_utterance(threshold):
    """Wait for speech, record until silence. Returns float32 audio or None."""
    block = int(SAMPLE_RATE * BLOCK_SECONDS)
    pre_roll = collections.deque(maxlen=10)   # keep 0.3s before speech starts
    frames = []
    started = False
    silent_blocks = 0
    start_time = 0.0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=block) as stream:
        while True:
            data, _ = stream.read(block)
            chunk = data[:, 0].copy()
            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if not started:
                pre_roll.append(chunk)
                if rms > threshold:
                    started = True
                    frames = list(pre_roll)
                    silent_blocks = 0
                    start_time = time.time()
            else:
                frames.append(chunk)
                silent_blocks = silent_blocks + 1 if rms < threshold else 0
                if silent_blocks * BLOCK_SECONDS > SILENCE_SECONDS:
                    break
                if time.time() - start_time > MAX_SECONDS:
                    break

    audio = np.concatenate(frames)
    if len(audio) / SAMPLE_RATE < MIN_SECONDS:
        return None
    return audio


def transcribe(model, audio):
    segments, _info = model.transcribe(
        audio, language=LANGUAGE, beam_size=1, vad_filter=True
    )
    return " ".join(s.text.strip() for s in segments).strip()


# ------------------------------ BRAIN --------------------------------------
def ask_brain(history):
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": history,
                "stream": False,
                "options": {"num_predict": 200, "temperature": 0.7},
            },
            timeout=180,
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return ("Ollama is not running. Please start Ollama and try again.")
    except Exception as e:  # noqa
        return f"Sorry, I had a problem thinking: {e}"


# ------------------------------ MOUTH --------------------------------------
def clean_for_speech(text):
    text = re.sub(r"[*_`#>~]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def speak_edge(text):
    import edge_tts
    import pygame

    voice = URDU_VOICE if re.search(r"[\u0600-\u06FF]", text) else ENGLISH_VOICE
    path = os.path.join(tempfile.gettempdir(), f"jarvis_{int(time.time()*1000)}.mp3")

    async def _make():
        await edge_tts.Communicate(text, voice).save(path)

    asyncio.run(_make())

    if not pygame.mixer.get_init():
        pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.05)
    pygame.mixer.music.unload()
    try:
        os.remove(path)
    except OSError:
        pass


def speak_offline(text):
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", 175)
    for v in engine.getProperty("voices"):
        if "zira" in v.name.lower() or "female" in v.name.lower():
            engine.setProperty("voice", v.id)
            break
    engine.say(text)
    engine.runAndWait()


def speak(text):
    text = clean_for_speech(text)
    if not text:
        return
    print(f"Jarvis: {text}")
    try:
        speak_edge(text)
    except Exception as e:  # no internet or edge-tts problem
        print(f"(online voice failed: {e} -> using offline voice)")
        try:
            speak_offline(text)
        except Exception as e2:
            print(f"(offline voice failed too: {e2})")


# ------------------------------ MAIN ---------------------------------------
def is_exit(text):
    t = text.lower().strip(" .!?،۔")
    return any(w in t for w in EXIT_WORDS)


def main():
    print("Loading speech model (first time downloads it, please wait)...")
    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    print("Measuring room noise... please stay quiet for 1 second.")
    threshold = calibrate_noise()

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    speak("Hello! Jarvis is ready. Bolo, main sun raha hoon.")

    try:
        while True:
            print("\nListening...")
            audio = record_utterance(threshold)
            if audio is None:
                continue

            text = transcribe(whisper, audio)
            if not text:
                continue
            print(f"You: {text}")

            if is_exit(text):
                speak("Goodbye!")
                break

            history.append({"role": "user", "content": text})
            reply = ask_brain(history)
            history.append({"role": "assistant", "content": reply})

            # keep only the last ~10 exchanges so the small model stays fast
            if len(history) > 21:
                history = [history[0]] + history[-20:]

            speak(reply)   # mic is not recording while Jarvis talks (no echo)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
