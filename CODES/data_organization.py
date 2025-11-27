"""
Organize dataset structure and metadata
---------------------------------------
Scans Mascellina dataset and builds a metadata.csv
with path, dog_name, label, and duration.
"""

import os
import pandas as pd
import librosa

def build_metadata(raw_dir, output_csv):
    rows = []
    for root, dirs, files in os.walk(raw_dir):
        label = os.path.basename(root)
        for f in files:
            if f.endswith(".wav"):
                file_path = os.path.join(root, f)
                dog_name = root.split(os.sep)[-2] if len(root.split(os.sep)) > 1 else "Unknown"
                try:
                    duration = librosa.get_duration(path=file_path)
                except:
                    duration = None
                rows.append({
                    "file_path": file_path,
                    "dog_name": dog_name,
                    "label": label,
                    "duration": duration
                })
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"✅ Metadata saved to {output_csv}, total samples: {len(df)}")
    return df
