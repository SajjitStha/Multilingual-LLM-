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

print("[INIT] Loading Hindi TTS (facebook/mms-tts-hin)...")
TTS_MODEL_ID = "facebook/mms-tts-hin"
tts_device = "cuda" if torch.cuda.is_available() else "cpu"

hi_tts_model = VitsModel.from_pretrained(TTS_MODEL_ID).to(tts_device)
hi_tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_ID)
print(f"[INIT] Hindi TTS loaded on {tts_device}")


# ===================== 2) HELPER FUNCTIONS =====================

def current_time_sentence_hindi() -> str:
    """
    Return current time in natural Hindi style, e.g.:
    'अभी चार बजकर पाँच मिनट हो गए हैं।'
    Digits are avoided because the TTS model often ignores them.
    """
    now = datetime.now()
    hour = now.hour
    minute = now.minute

    # Convert to 12-hour format
    hour12 = hour % 12
    if hour12 == 0:
        hour12 = 12

    hi_hour_words = {
        1: "एक",
        2: "दो",
        3: "तीन",
        4: "चार",
        5: "पाँच",
        6: "छह",
        7: "सात",
        8: "आठ",
        9: "नौ",
        10: "दस",
        11: "ग्यारह",
        12: "बारह",
    }

    minute_words = {
        0: "शून्य",
        1: "एक",
        2: "दो",
        3: "तीन",
        4: "चार",
        5: "पाँच",
        6: "छह",
        7: "सात",
        8: "आठ",
        9: "नौ",
        10: "दस",
        11: "ग्यारह",
        12: "बारह",
        13: "तेरह",
        14: "चौदह",
        15: "पंद्रह",
        16: "सोलह",
        17: "सत्रह",
        18: "अठारह",
        19: "उन्नीस",
        20: "बीस",
        21: "इक्कीस",
        22: "बाईस",
        23: "तेइस",
        24: "चौबीस",
        25: "पच्चीस",
        26: "छब्बीस",
        27: "सत्ताईस",
        28: "अट्ठाईस",
        29: "उनतीस",
        30: "तीस",
        31: "इकतीस",
        32: "बतीस",
        33: "तैंतीस",
        34: "चौंतीस",
        35: "पैंतीस",
        36: "छत्तीस",
        37: "सैंतीस",
        38: "अड़तीस",
        39: "उनतालीस",
        40: "चालीस",
        41: "इकतालीस",
        42: "बयालीस",
        43: "तैंतालीस",
        44: "चवालीस",
        45: "पैंतालीस",
        46: "छियालिस",
        47: "सैंतालीस",
        48: "अड़तालीस",
        49: "उनचास",
        50: "पचास",
        51: "इक्यावन",
        52: "बावन",
        53: "तिरेपन",
        54: "चौवन",
        55: "पचपन",
        56: "छप्पन",
        57: "सत्तावन",
        58: "अट्ठावन",
        59: "उनसठ",
    }

    h_word = hi_hour_words.get(hour12, str(hour12))

    if minute == 0:
        return f"अभी {h_word} बजे हैं।"

    m_word = minute_words.get(minute, str(minute))
    return f"अभी {h_word} बजकर {m_word} मिनट हो गए हैं।"



def looks_like_garbage(text: str) -> bool:
    """
    Relaxed noise guard: only treat completely empty text as garbage.
    """
    return len(text.strip()) == 0


def whisper_transcribe_hindi(wav_path: str) -> str:
    """
    Use Whisper to convert speech -> Hindi text.
    """
    print(f"[STT] Transcribing file: {wav_path}")
    result = whisper_model.transcribe(
        wav_path,
        task="transcribe",
        language="hi",   # Hindi
        temperature=0.0,
        best_of=3,
        beam_size=5,
    )
    text = result["text"].strip()
    print(f"[STT] Transcription: {text}")
    return text


def generate_hindi_reply(user_text: str) -> str:
    """
    Single LLM call that:
    - always answers in Hindi (Devanagari),
    - uses the current time sentence when the question is about time (via prompt),
    - otherwise answers logically but conservatively (no wild guessing).
    """
    if looks_like_garbage(user_text):
        print("[RULE] Empty or garbage text, asking user to repeat.")
        return "मुझे तुम्हारी बात ठीक से सुनाई नहीं दी, ज़रा फिर से साफ़-साफ़ बोलो।"

    time_sentence = current_time_sentence_hindi()

    # Use smaller Gemma model for stability and speed
    model_name = "gemma3:1b"

    # Few-shot style prompt with examples to guide time and identity answers
    prompt = (
        "तुम एक मददगार, ईमानदार और सावधान हिंदी वॉइस असिस्टेंट हो। "
        "तुम्हें हमेशा सिर्फ़ हिंदी (देवनागरी) में ही जवाब देना है। "
        "दूसरी भाषा में मत लिखो।\n\n"
        f"अभी का असली समय (यह तथ्य है, इसे गलत नहीं मानना):\n'{time_sentence}'\n\n"
        "नीचे कुछ उदाहरण हैं:\n"
        "उदाहरण 1:\n"
        "उपयोगकर्ता: अभी कितने बजे हैं?\n"
        f"सहायक: {time_sentence}\n\n"
        "उदाहरण 2:\n"
        "उपयोगकर्ता: तुम कौन हो?\n"
        "सहायक: मैं तुम्हारा ऑफलाइन हिंदी वॉइस असिस्टेंट हूँ। "
        "मैं तुम्हारी बातें सुनकर हिंदी में जवाब देता हूँ।\n\n"
        "अब निर्देश:\n"
        "1) अगर उपयोगकर्ता का सवाल समय के बारे में हो (जैसे: अभी कितने बजे हैं, "
        "   अभी कितना समय हुआ है, आदि) तो ऊपर दिए गए असली समय को "
        "   उपयोग करके छोटा और साफ़ जवाब दो।\n"
        "2) अगर सवाल समय के बारे में नहीं है, तो सामान्य सवाल की तरह समझ कर "
        "   छोटा, स्पष्ट और तार्किक हिंदी में जवाब दो।\n"
        "3) अगर तुम्हें किसी विषय के बारे में पक्का पता नहीं है (जैसे इतिहास, गाना, तथ्य), "
        "   तो झूठा या कल्पना वाला उत्तर मत बनाओ। "
        "   ऐसे में ईमानदारी से कहो: 'मुझे इस बारे में ठीक से जानकारी नहीं है।'\n"
        "4) अगर ASR से आया वाक्य बहुत अजीब, अधूरा या समझ में न आने वाला हो, "
        "   तो खुद से अर्थ मत लगाओ और कहो: "
        "   'मुझे ठीक से समझ नहीं आया, ज़रा और साफ़ बोलो या फिर से पूछो।'\n\n"
        f"उपयोगकर्ता का वाक्य (ASR आउटपुट, इसमें गलतियाँ हो सकती हैं):\n'{user_text}'\n\n"
        "अब ऊपर दिए गए नियमों को मानते हुए, सहायक का जवाब (सिर्फ़ हिंदी में):"
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
                    "temperature": 0.15,   # low randomness = less hallucination
                    "top_p": 0.7,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_reply = data.get("response", "").strip()
    except Exception as e:
        print(f"[LLM] Error calling Ollama: {e}")
        return "माफ़ करना, अभी मैं ठीक से जवाब नहीं दे पाया। थोड़ी देर बाद फिर से कोशिश करना।"

    # Clean weird symbols (keep Devanagari + digits + basic punctuation)
    allowed_pattern = r"[^0-9 \u0900-\u097F।,\.?!]"
    reply_clean = re.sub(allowed_pattern, " ", raw_reply)
    reply_clean = re.sub(r"\s+", " ", reply_clean).strip()

    if not reply_clean:
        reply_clean = "मुझे ठीक से समझ नहीं आया, ज़रा और साफ़ बोलकर फिर से पूछो।"

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
        raise RuntimeError(
            "कोई ऑडियो कैप्चर नहीं हुआ; माइक के पास थोड़ा साफ़ और ज़ोर से बोलकर फिर कोशिश करो।"
        )

    audio = np.concatenate(audio_frames, axis=0)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, audio, FS)
    print(f"[REC] Saved to temp file: {tmp.name}")
    return tmp.name


def speak_hindi(text: str):
    print(f"[TTS] Synthesizing Hindi audio for: {text!r}")
    try:
        inputs = hi_tts_tokenizer(text, return_tensors="pt").to(tts_device)
        with torch.no_grad():
            outputs = hi_tts_model(**inputs).waveform  # [1, T]
        waveform = outputs.squeeze().cpu().numpy()
        sr = hi_tts_model.config.sampling_rate

        sd.stop()
        sd.play(waveform, sr)
        sd.wait()
        print("[TTS] Done.")
    except Exception as e:
        print(f"[TTS] Error (Hindi VITS): {e}")
        append_status("हिंदी आवाज़ निकालने में दिक्कत आई, अभी सिर्फ़ टेक्स्ट दिखेगा।")


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
            append_status("रिकॉर्ड हो रहा है... बोलने के बाद Stop दबाओ।")
        except Exception as e:
            print(f"[REC] Error starting recording: {e}")
            append_status(f"रिकॉर्ड शुरू करने में समस्या आई: {e}")
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
        append_status("रिकॉर्ड रोक दिया, प्रोसेस हो रहा है...")
        wav_path = stop_recording_and_save()

        append_status("Whisper तुम्हारी आवाज़ सुन रहा है...")
        user_text = whisper_transcribe_hindi(wav_path)
        append_chat("तुम", user_text)

        append_status("सहायक सोच रहा है...")
        reply = generate_hindi_reply(user_text)
        append_chat("सहायक", reply)

        append_status("हिंदी आवाज़ निकाल रहा हूँ...")
        speak_hindi(reply)

        append_status("तैयार हूँ 🎧 फिर से Speak दबाकर बात कर सकते हो।")
    except Exception as e:
        print(f"[ERROR] {e}")
        append_status(f"कोई त्रुटि आई: {e}")
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
            print(f"[REC] Deleted temp file: {wav_path}")
        busy = False
        recording = False
        speak_button.config(state="normal")


# ===================== 5) BUILD TKINTER UI =====================

root = tk.Tk()
root.title("हिंदी Voice Assistant (Offline - Whisper + Gemma3 + Hindi TTS)")
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

status_label = tk.Label(
    root,
    text="तैयार हूँ। Speak दबाकर बोलो, फिर Stop दबाकर रोक दो।",
    anchor="w"
)
status_label.pack(fill=tk.X, padx=10, pady=5)

append_chat("सिस्टम", "नमस्ते! मैं तुम्हारा ऑफलाइन हिंदी वॉइस असिस्टेंट हूँ। मुझसे हिंदी में बात करो।")

print("[INIT] UI ready.")
root.mainloop()
