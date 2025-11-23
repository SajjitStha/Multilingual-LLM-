import os
import re
import tempfile
from datetime import datetime

import tkinter as tk
from tkinter.scrolledtext import ScrolledText

import sounddevice as sd
import soundfile as sf
import whisper
import torch
import requests
import numpy as np

from transformers import VitsModel, AutoTokenizer

# ===================== 1) GLOBAL CONFIG & MODELS =====================

FS = 16000  # sample rate for recording

# Global state for push-to-talk
recording = False
busy = False            # pipeline running flag (STT + LLM + TTS)
audio_frames = []
audio_stream = None

print("[INIT] Loading Whisper model (medium for better accuracy)...")
# You can change to "small" for speed, "medium" for better accuracy.
whisper_model = whisper.load_model("medium")

print("[INIT] Loading Nepali TTS (tuskbyte/nepali_male_v1)...")
TTS_MODEL_ID = "tuskbyte/nepali_male_v1"
tts_device = "cuda" if torch.cuda.is_available() else "cpu"

np_tts_model = VitsModel.from_pretrained(TTS_MODEL_ID).to(tts_device)
np_tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_ID)
print(f"[INIT] Nepali TTS loaded on {tts_device}")


# ===================== 2) HELPER FUNCTIONS =====================

def current_time_sentence_nepali() -> str:
    """
    Return current time in natural Nepali style, e.g.:
    'अहिले चार बजेर पाँच मिनेट भएको छ।'
    """
    now = datetime.now()
    hour = now.hour
    minute = now.minute

    # Convert to 12-hour format
    hour12 = hour % 12
    if hour12 == 0:
        hour12 = 12

    nep_hour_words = {
        1: "एक",
        2: "दुई",
        3: "तीन",
        4: "चार",
        5: "पाँच",
        6: "छ",
        7: "सात",
        8: "आठ",
        9: "नौ",
        10: "दश",
        11: "एघार",
        12: "बाह्र",
    }

    h_word = nep_hour_words.get(hour12, str(hour12))

    nep_min_words = {
        0: "सुन्य",
        1: "एक",
        2: "दुई",
        3: "तीन",
        4: "चार",
        5: "पाँच",
        6: "छ",
        7: "सात",
        8: "आठ",
        9: "नौ",
        10: "दश",
        11: "एघार",
        12: "बाह्र",
    }

    if minute == 0:
        return f"अहिले {h_word} बजेको छ।"
    else:
        m_word = nep_min_words.get(minute, str(minute))
        return f"अहिले {h_word} बजेर {m_word} मिनेट भएको छ।"


def looks_like_garbage(text: str) -> bool:
    """
    Very simple noise guard: if text is mostly repetition, don't send
    it to LLM; ask user to repeat.
    """
    words = text.split()
    if not words:
        return True

    from collections import Counter
    counts = Counter(words)
    most_common_word, freq = counts.most_common(1)[0]
    if freq >= 5:
        print(f"[NOISE] Repetitive text: '{most_common_word}' x {freq}")
        return True

    unique_chars = len(set(text))
    if len(text) > 40 and unique_chars < 5:
        print("[NOISE] Very low character variety in long text.")
        return True

    return False


def whisper_transcribe_hindi(wav_path: str) -> str:
    """
    Use Whisper to convert speech -> Hindi text.
    (language='hi' instead of 'ne')
    """
    print(f"[STT] Transcribing file: {wav_path}")
    result = whisper_model.transcribe(
        wav_path,
        task="transcribe",
        language="hi",   # <-- CHANGED: Hindi now
        temperature=0.0,
        best_of=3,
        beam_size=5,
    )
    text = result["text"].strip()
    print(f"[STT] Transcription: {text}")
    return text


def generate_nepali_reply(user_text: str) -> str:
    """
    Single LLM call that:
    - always answers in Nepali,
    - uses the provided 'current time' sentence if user is asking time,
    - otherwise just answers logically as a normal assistant.
    """

    if looks_like_garbage(user_text):
        print("[RULE] Noise detected, asking user to repeat.")
        return "मलाई तिम्रो कुरा राम्ररी सुनिनँ, फेरि एकचोटि बिस्तारै भन न।"

    time_sentence = current_time_sentence_nepali()

    model_name = "llama3.2:3b"  # or "llama3.2:1b" for faster if you have it pulled
    # you can change this to "gemma3:4b" later if you want

    prompt = (
        "तिमी एक सहयोगी, तार्किक र सचेत नेपाली आवाज सहायक हौ। "
        "तिमीले सधैं नेपाली भाषामा मात्र जवाफ दिनुपर्छ।\n\n"
        f"हालको वास्तविक समय यस्तो छ:\n'{time_sentence}'\n\n"
        "यदि प्रयोगकर्ताको प्रश्न समयबारे छ (जस्तै: अहिले कति बजे हो, कति बज्यो, "
        "अहिलेको समय कति भयो आदि) भने, माथिको वाक्यमा दिएको समय प्रयोग गरेर "
        "ठ्याक्कै जवाफ देऊ।\n\n"
        "यदि प्रश्न समयबारे होइन भने, अरू सामान्य प्रश्न-जस्तो ठानेर "
        "तार्किक, छोटो र स्पष्ट नेपालीमा जवाफ देऊ।\n\n"
        "यदि वाक्य अस्पष्ट छ भने, इमानदारीपूर्वक 'मलाई ठीकसँग बुझिएन, "
        "फेरि अलि स्पष्ट भएर सोध न।' भनेर जवाफ देऊ।\n\n"
        f"प्रयोगकर्ताको वाक्य (ASR आउटपुट, गल्ती हुन सक्छ - हिन्दीमा पनि हुन सक्छ):\n'{user_text}'\n\n"
        "अब, सहायकको जवाफ (नेपालीमा मात्र):"
    )

    print("[LLM] Calling Ollama...")
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.8,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_reply = data.get("response", "").strip()
    except Exception as e:
        print(f"[LLM] Error calling Ollama: {e}")
        return "माफ गर, अहिले म राम्रोसँग सोच्न सकिनँ। फेरि प्रयास गर न।"

    # Clean weird symbols (stick to Devanagari + basic punctuation)
    allowed_pattern = r"[^ \u0900-\u097F\u0966-\u096F।,\.?!]"
    reply_clean = re.sub(allowed_pattern, " ", raw_reply)
    reply_clean = re.sub(r"\s+", " ", reply_clean).strip()

    if not reply_clean:
        reply_clean = "मलाई ठीकसँग बुझिएन, फेरि अलि स्पष्ट भएर सोध्नु न।"

    print(f"[LLM] Raw reply: {raw_reply}")
    print(f"[LLM] Clean reply: {reply_clean}")
    return reply_clean


# ===================== 3) PUSH-TO-TALK RECORDING =====================

def start_recording():
    global recording, audio_frames, audio_stream
    print("[REC] Start recording (push-to-talk)...")
    audio_frames = []

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[REC] Status: {status}")
        audio_frames.append(indata.copy())

    audio_stream = sd.InputStream(
        samplerate=FS,
        channels=1,
        callback=callback
    )
    audio_stream.start()
    recording = True


def stop_recording_and_save() -> str:
    global recording, audio_stream, audio_frames

    print("[REC] Stop recording.")
    if audio_stream is not None:
        audio_stream.stop()
        audio_stream.close()
        audio_stream = None

    recording = False

    if not audio_frames:
        raise RuntimeError("No audio captured; try speaking a bit louder or closer to the mic.")

    audio = np.concatenate(audio_frames, axis=0)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, audio, FS)
    print(f"[REC] Saved to temp file: {tmp.name}")
    return tmp.name


def speak_nepali(text: str):
    print(f"[TTS] Synthesizing Nepali audio for: {text!r}")
    try:
        inputs = np_tts_tokenizer(text, return_tensors="pt").to(tts_device)
        with torch.no_grad():
            outputs = np_tts_model(**inputs).waveform  # [1, T]
        waveform = outputs.squeeze().cpu().numpy()
        sr = np_tts_model.config.sampling_rate

        sd.stop()
        sd.play(waveform, sr)
        sd.wait()
        print("[TTS] Done.")
    except Exception as e:
        print(f"[TTS] Error (Nepali VITS): {e}")
        append_status("नेपाली आवाज निकाल्न समस्या आयो, अहिले टेक्स्ट मात्र देखाइन्छ।")


# ===================== 4) GUI CALLBACKS =====================

def append_chat(prefix: str, text: str):
    chat_box.config(state="normal")
    chat_box.insert(tk.END, f"{prefix}: {text}\n")
    chat_box.insert(tk.END, "-" * 60 + "\n")
    chat_box.see(tk.END)
    chat_box.config(state="disabled")


def append_status(text: str):
    status_label.config(text=text)


def on_speak_button():
    """
    Push-to-talk:

    - If busy with previous pipeline, ignore extra clicks.
    - If not recording and not busy => start recording.
    - If recording and not busy => stop and process.
    """
    global recording, busy

    if busy:
        print("[UI] Click ignored because pipeline is busy.")
        return

    if not recording:
        # Start recording
        try:
            start_recording()
            speak_button.config(text="⏹ Stop")
            append_status("रेकर्ड हुँदैछ... बोलिसकेपछि Stop थिच।")
        except Exception as e:
            print(f"[REC] Error starting recording: {e}")
            append_status(f"रेकर्ड सुरु गर्न समस्या आयो: {e}")
            recording = False
            speak_button.config(text="🎙️ Speak")
        return

    # Currently recording -> stop and process
    print("[UI] Stop button clicked, processing audio...")
    busy = True
    speak_button.config(state="disabled")
    speak_button.config(text="🎙️ Speak")

    wav_path = None
    try:
        append_status("रेकर्ड रोकियो, प्रक्रिया हुँदैछ...")
        wav_path = stop_recording_and_save()

        append_status("Whisper ले (हिन्दीमा) सुन्दैछ...")
        user_text = whisper_transcribe_hindi(wav_path)
        append_chat("तिमी", user_text)

        append_status("सहायकले सोच्दै...")
        reply = generate_nepali_reply(user_text)
        append_chat("सहायक", reply)

        append_status("नेपाली आवाज निकाल्दै...")
        speak_nepali(reply)

        append_status("तयार छ 🎧 फेरि Speak थिचेर बोल।")
    except Exception as e:
        print(f"[ERROR] {e}")
        append_status(f"त्रुटि आयो: {e}")
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
            print(f"[REC] Deleted temp file: {wav_path}")
        busy = False
        recording = False
        speak_button.config(state="normal")


# ===================== 5) BUILD TKINTER UI =====================

root = tk.Tk()
root.title("नेपाली Voice Assistant (Offline - Ollama + Nepali TTS)")
root.geometry("700x500")

chat_box = ScrolledText(root, wrap=tk.WORD, state="disabled", font=("Nirmala UI", 11))
chat_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

controls_frame = tk.Frame(root)
controls_frame.pack(fill=tk.X, padx=10, pady=5)

speak_button = tk.Button(
    controls_frame,
    text="🎙️ Speak",
    font=("Segoe UI", 12, "bold"),
    command=on_speak_button,
)
speak_button.pack(side=tk.LEFT)

status_label = tk.Label(root, text="तयार छ। Speak थिचेर बोल, फेरि Stop थिचेर रोक।", anchor="w")
status_label.pack(fill=tk.X, padx=10, pady=5)

append_chat("सिस्टम", "नमस्ते! म तिम्रो अफलाइन भ्वाइस सहायक हुँ। अहिले म तिम्रो आवाज हिन्दीमा बुझ्छु र नेपालीमा जवाफ दिन्छु।")

print("[INIT] UI ready.")
root.mainloop()
