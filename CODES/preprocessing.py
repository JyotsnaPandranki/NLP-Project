"""
Audio preprocessing
-------------------
Noise reduction, normalization, and optional segmentation.
"""

import os
from pydub import AudioSegment
import librosa
import numpy as np
import noisereduce as nr

def preprocess_audio(file_path, sample_rate=16000, segment_len=2000):
    # Load audio
    y, sr = librosa.load(file_path, sr=sample_rate)
    # Noise reduction
    y = nr.reduce_noise(y=y, sr=sr)
    # Normalize
    y = librosa.util.normalize(y)
    # Convert to pydub for segmentation
    audio = AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    # Segment to fixed length (e.g. 2 s = 2000 ms)
    segments = []
    for i in range(0, len(audio), segment_len):
        seg = audio[i:i+segment_len]
        if len(seg) == segment_len:
            segments.append(np.array(seg.get_array_of_samples()).astype(np.float32) / 32768.0)
    return segments
 