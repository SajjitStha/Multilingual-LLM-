import torch
import sounddevice as sd
from transformers import VitsModel, AutoTokenizer

MODEL_ID = "tuskbyte/nepali_male_v1"

print("Loading Nepali TTS model (tuskbyte/nepali_male_v1)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = VitsModel.from_pretrained(MODEL_ID).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Test sentence in Nepali
text = "नमस्ते, म नेपालीमा कुरा गर्दैछु।"

print("Tokenizing...")
inputs = tokenizer(text, return_tensors="pt").to(device)

print("Synthesizing waveform...")
with torch.no_grad():
    outputs = model(**inputs).waveform  # [1, T]

waveform = outputs.squeeze().cpu().numpy()
sr = model.config.sampling_rate

print(f"Playing audio at {sr} Hz...")
sd.stop()
sd.play(waveform, sr)
sd.wait()
print("Done.")
