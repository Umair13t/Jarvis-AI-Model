# Jarvis - Step 1: Voice Loop

Tum bolte ho -> Jarvis sunta hai -> sochta hai -> bol kar jawab deta hai.
Sab free hai. Awaz sunna aur dimaag tumhare laptop par chalte hain.
Sirf bolne wali awaz (edge-tts) internet use karti hai. Internet na ho to offline Windows voice chal jati hai.

## Setup (Windows 10)

1. Ollama install ho aur model download ho:
   ```
   ollama pull llama3.2:3b
   ```
   (Ollama background mein chal raha hona chahiye. Taskbar mein icon dekho.)

2. Is folder mein terminal kholo (VS Code > Terminal > New Terminal), phir:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   Agar PowerShell activate par error de to ye pehle chalao:
   ```
   Set-ExecutionPolicy -Scope Process Bypass
   ```

3. Chalao:
   ```
   python jarvis.py
   ```
   Pehli baar Whisper model download hoga (thora time lagega).

## Use
- "Listening..." likha aaye to bolo.
- Band karne ke liye bolo: "exit" ya "bye jarvis" ya "band karo". Ya Ctrl+C.

## Settings (jarvis.py ke upar)
- `WHISPER_MODEL = "tiny"` agar slow lage. `"base"` zyada sahi sunta hai.
- `LANGUAGE = "ur"` ya `"en"` agar auto-detect galat ho.
- `OLLAMA_MODEL` badal sakte ho (bara model = zyada slow).
- Agar Jarvis chup chaap sunta hi na ho ya har awaz par chal parhe to `calibrate_noise` ki threshold dekho.

## Aglay steps
2. Memory (kal bhi yaad rakhe)  3. Laptop control  4. Screen awareness  5. Webcam
