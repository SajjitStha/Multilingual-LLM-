# Offline Nepali & Hindi Voice Assistant 🎙️

Desktop apps that let you talk in **Nepali or Hindi**, and get a spoken reply back — all **offline** using:

- **Whisper** → speech-to-text  
- **Ollama (Gemma3)** → local LLM  
- **TTS models** → Nepali / Hindi voice

Files:
- `nepali_voice_app.py` – Nepali assistant  
- `hindi_voice_app.py` – Hindi assistant  

---

## 1. What you need

- **Windows 10/11 (64-bit)** with a mic  
- **Python 3.10+**  
- **FFmpeg** in PATH (check with `ffmpeg -version`)  
- **Ollama** installed & running → https://ollama.com/download  
- Basic Python deps:

```bash
pip install openai-whisper transformers sentencepiece sounddevice soundfile torch requests
Models (downloaded automatically on first run)
Whisper medium

Nepali TTS: tuskbyte/nepali_male_v1

Hindi TTS: facebook/mms-tts-hin

LLM via Ollama: gemma3:1b

Pull the LLM:

bash
Copy code
ollama pull gemma3:1b
2. Run the Hindi assistant
bash
Copy code
python hindi_voice_app.py
A small window opens.

Click 🎙️ Speak, say something in Hindi (e.g. “तुम कौन हो?”, “अभी कितने बजे हैं?”).

Click ⏹ Stop.

It will:

show the recognized text,

show the reply in Hindi,

speak the reply.

3. Run the Nepali assistant
bash
Copy code
python nepali_voice_app.py
Same flow, but you speak in Nepali, and it replies + speaks in Nepali.

4. Notes
First run may be slow while models download.

Everything runs locally after that (offline).

Small models → answers are simple, not perfect.
