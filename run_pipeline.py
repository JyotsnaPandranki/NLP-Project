import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
from sklearn.model_selection import train_test_split
from joblib import dump

from src.data_organization import build_metadata
from src.preprocessing import preprocess_audio
from src.feature_engineering import extract_features
from src.train_model import train_cnn
from src.evaluate_model import evaluate

RAW_DIR = "data/raw"
META_CSV = "data/metadata.csv"

os.makedirs("models", exist_ok=True)

print("\n🐶 Starting Dog Voice Translation Pipeline...\n")

# Stage 1 — Metadata
meta = build_metadata(RAW_DIR, META_CSV)

# Stage 2–3 — Preprocess + Extract Features
X, y = [], []

for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Processing files"):
    try:
        segments = preprocess_audio(row.file_path)
        for seg in segments:
            feats = extract_features(seg)
            X.append(feats)
            y.append(row.label)
    except Exception as e:
        print(f"⚠️ Error {row.file_path}: {e}")

X = np.array(X)
y = np.array(y)

# Save full labeled dataset for evaluation reuse
np.save("models/X_full.npy", X)
np.save("models/y_full.npy", y)

# Split into train/test before encoding labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save test for later evaluation
np.save("models/X_test.npy", X_test)
np.save("models/y_test.npy", y_test)

# Stage 4 — Train
model, encoder = train_cnn(X_train, y_train)

print("\n🔥 Training Done — Evaluating now...\n")

# Stage 5 — Evaluate
X_test = X_test[..., np.newaxis]
evaluate(model, encoder, X_test, encoder.transform(y_test))

print("\n🎉 Pipeline completed successfully!")
