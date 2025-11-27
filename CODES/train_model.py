import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

def train_cnn(X, y, model_path="models/dog_voice_classifier.h5"):

    # Save encoder using original string labels
    encoder = LabelEncoder()
    y_int = encoder.fit_transform(y)

    np.save("models/label_classes.npy", encoder.classes_)
    joblib.dump(encoder, "models/label_encoder.pkl")

    X = X[..., np.newaxis]  # CNN channel dimension

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_int, test_size=0.2, random_state=42, stratify=y_int
    )

    num_classes = len(encoder.classes_)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=X_train.shape[1:]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)
    ckpt = ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy')

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        callbacks=[es, ckpt],
        verbose=1
    )

    print("🔥 Model + Encoder saved!")
    return model, encoder
