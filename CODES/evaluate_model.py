import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(model, encoder, X_test, y_test):

    print("\n📊 Running predictions...")
    pred_probs = model.predict(X_test)
    y_pred = np.argmax(pred_probs, axis=1)

    print("\n📈 Classification Report:\n")
    print(classification_report(
        y_test, y_pred, target_names=encoder.classes_, digits=3
    ))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_,
                cmap="Purples")
    plt.title("Confusion Matrix - Dog Voice Classifier")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png")
    plt.show()

    print("🎯 Confusion matrix saved to: models/confusion_matrix.png")
