# predict_realtime.py
import os, queue, time, sys
import numpy as np
import sounddevice as sd
import librosa
from tensorflow.keras.models import load_model

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from feature_engineering import extract_features
from translate_bark import translate_label

MODEL_PATH = "models/dog_voice_classifier.h5"
LABELS_PATH = "models/label_classes.npy"
TARGET_SR = 16000

print("🔧 Loading model…")
model = load_model(MODEL_PATH)
labels = np.load(LABELS_PATH)
print(f"✅ Model loaded ({len(labels)} classes).")

# mic parameters
CHUNK_SEC = 0.1
START_THRESH = 0.015
STOP_THRESH = 0.008
MAX_CLIP_SEC = 3.0
SILENCE_TAIL_SEC = 0.7
q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    q.put(indata.copy())

def rms(x):
    return np.sqrt(np.mean(x**2) + 1e-12)

def collect_clip(stream_sr):
    started, buf, last_above = False, [], time.time()
    while True:
        x = q.get().astype(np.float32).squeeze()
        e = rms(x)
        now = time.time()
        if not started and e > START_THRESH:
            started, buf = True, [x]
            last_above = now
        elif started:
            buf.append(x)
            if e > STOP_THRESH:
                last_above = now
            if now - last_above > SILENCE_TAIL_SEC or len(buf) > stream_sr * MAX_CLIP_SEC:
                break
    clip = np.concatenate(buf) if buf else np.zeros(int(stream_sr*0.5), np.float32)
    if stream_sr != TARGET_SR:
        clip = librosa.resample(clip, orig_sr=stream_sr, target_sr=TARGET_SR)
    clip /= np.max(np.abs(clip)) + 1e-9
    return clip

def predict_clip(y):
    feats = extract_features(y, sr=TARGET_SR)
    feats = np.expand_dims(feats, axis=(0, -1))
    probs = model.predict(feats, verbose=0)[0]
    idx = np.argmax(probs)
    return labels[idx], probs[idx]

def main():
    print("\n🎤 Real-time Dog → Text Translator\n(Press Ctrl+C to stop)\n")
    device = sd.query_devices(kind='input')
    sr = int(device['default_samplerate'])
    blocksize = int(sr * CHUNK_SEC)
    with sd.InputStream(channels=1, samplerate=sr, blocksize=blocksize, callback=audio_callback):
        while True:
            try:
                clip = collect_clip(sr)
                if len(clip) < TARGET_SR * 0.3:
                    continue
                label, conf = predict_clip(clip)
                phrase = translate_label(label, conf)
                print(f"👉 {label} ({conf:.2f})  |  💬 {phrase}")
            except KeyboardInterrupt:
                print("\n👋 Stopped.")
                break

if __name__ == "__main__":
    main()
