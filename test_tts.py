import pyttsx3

engine = pyttsx3.init()

print("=== Available voices ===")
for i, v in enumerate(engine.getProperty("voices")):
    print(f"{i}: id={v.id!r}, name={v.name!r}")

engine.setProperty("volume", 1.0)
engine.setProperty("rate", 180)

print("\nTrying default voice…")
engine.say("Hello, this is a test voice from Python.")
engine.runAndWait()
print("Done.")
