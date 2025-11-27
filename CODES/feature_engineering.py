"""
Feature extraction for each segment
-----------------------------------
Generates MFCC, spectral contrast, chroma, and RMS energy features.
"""

import librosa
import numpy as np

def extract_features(y, sr=16000, n_mfcc=40, max_len=100):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)

    # Combine and pad/truncate
    features = np.vstack([mfcc, spec_contrast, chroma, rms])
    if features.shape[1] < max_len:
        pad_width = max_len - features.shape[1]
        features = np.pad(features, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        features = features[:, :max_len]
    return features
