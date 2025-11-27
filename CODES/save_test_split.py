import numpy as np
from sklearn.model_selection import train_test_split

print("📥 Loading full dataset...")

# Load the full dataset that was already used for training
X = np.load("models/X_full.npy")
y = np.load("models/y_full.npy")

print(f"Loaded full dataset → X: {X.shape}, y: {y.shape}")

# Create test split again (same as before)
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save test data for evaluation
np.save("models/X_test.npy", X_test)
np.save("models/y_test.npy", y_test)

print("✔ Test split saved!")
print("📁 Saved: models/X_test.npy and models/y_test.npy")
print("Now run: python src/evaluate_model.py")
