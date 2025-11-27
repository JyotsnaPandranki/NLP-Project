import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(model, encoder, X_test, y_test):
    # Predictions
    pred_probs = model.predict(X_test)
    y_pred = np.argmax(pred_probs, axis=1)

    # Fix: if labels already integers, do not argmax
    if len(y_test.shape) == 1:
        y_true = y_test
    else:
        y_true = np.argmax(y_test, axis=1)

    print("\nClassification Report:\n")
    print(classification_report(
        y_true, y_pred, target_names=encoder.classes_, digits=3
    ))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — Dog Voice Classifier")
    plt.tight_layout()
    plt.show()
