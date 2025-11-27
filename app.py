import os, io, sys
import numpy as np
import librosa
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

sys.path.append("src")
from preprocessing import preprocess_audio
from feature_engineering import extract_features
from translate_bark import translate_label

MODEL_PATH = "models/dog_voice_classifier.h5"
LABELS_PATH = "models/label_classes.npy"
TARGET_SR = 16000

@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH)
    labels = np.load(LABELS_PATH, allow_pickle=True)
    return model, labels

model, labels = load_assets()

st.set_page_config(page_title="Dog Bark Translator", page_icon="🐶")
st.title("🐶 Dog Voice → Human Text Translator")

uploaded = st.file_uploader("Upload a bark (.wav)", type=["wav"])

def predict_from_segments(file):
    segments = preprocess_audio(file)
    predictions = []

    for seg in segments:
        feats = extract_features(seg)
        feats = feats[np.newaxis, ..., np.newaxis]
        probs = model.predict(feats, verbose=0)[0]
        predictions.append(probs)

    avg_probs = np.mean(predictions, axis=0)
    idx = np.argmax(avg_probs)
    return labels[idx], avg_probs


if uploaded:
    data = uploaded.read()

    st.audio(data, format="audio/wav")
    filepath = "temp.wav"
    with open(filepath, "wb") as f:
        f.write(data)

    label, probs = predict_from_segments(filepath)
    phrase = translate_label(label, probs[np.argmax(probs)])

    st.success(f"Detected: **{label}** | Confidence: {probs.max():.2f}")
    st.info(f"💬 {phrase}")

    y, sr = librosa.load(filepath, sr=TARGET_SR)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(y)
    ax.set_title("Waveform")
    ax.set_yticks([])
    st.pyplot(fig)

    st.subheader("Top-5 Predictions")
    top_idx = np.argsort(probs)[::-1][:5]
    for i in top_idx:
        st.write(f"**{labels[i]}** — {probs[i]:.2f}")
else:
    st.caption("Upload a dog bark audio to decode 🐕💬")
