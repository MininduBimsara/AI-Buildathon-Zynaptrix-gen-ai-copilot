"""
autoencoder_model.py — Dense (feedforward) Autoencoder for sensor anomaly detection.

Architecture:
    Input(n_features)
        ↓  Dense(16, relu)
        ↓  Dense(8,  relu)   ← encoder
        ↓  Dense(4,  relu)   ← latent bottleneck
        ↓  Dense(8,  relu)   ← decoder
        ↓  Dense(16, relu)
        ↓  Dense(n_features, linear)  ← reconstruction

The model is trained ONLY on normal-state data.
Reconstruction error (MSE) on unseen data is used as the anomaly score.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def build_autoencoder(n_features: int = 5,
                      latent_dim: int = 4,
                      dropout_rate: float = 0.1) -> Model:
    """
    Build and compile a Dense Autoencoder.

    Args:
        n_features:   Number of input/output features (sensor columns).
        latent_dim:   Bottleneck dimension.
        dropout_rate: Dropout regularization rate.

    Returns:
        Compiled Keras Model.
    """
    inputs = Input(shape=(n_features,), name="encoder_input")

    # ── Encoder ──────────────────────────────────
    x = Dense(16, activation="relu", name="enc_1")(inputs)
    x = Dropout(dropout_rate)(x)
    x = Dense(8,  activation="relu", name="enc_2")(x)
    x = Dropout(dropout_rate)(x)
    latent = Dense(latent_dim, activation="relu", name="latent")(x)

    # ── Decoder ──────────────────────────────────
    x = Dense(8,  activation="relu", name="dec_1")(latent)
    x = Dropout(dropout_rate)(x)
    x = Dense(16, activation="relu", name="dec_2")(x)
    outputs = Dense(n_features, activation="linear", name="reconstruction")(x)

    model = Model(inputs, outputs, name="DenseAutoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse",
                  metrics=["mae"])
    return model


def get_callbacks(patience: int = 10) -> list:
    """Return standard training callbacks."""
    return [
        EarlyStopping(monitor="val_loss", patience=patience,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1),
    ]


def reconstruction_error(model: Model, X: np.ndarray) -> np.ndarray:
    """
    Compute per-sample Mean Squared Error between input and reconstruction.

    Returns:
        1-D array of shape (n_samples,) with anomaly scores.
    """
    X_pred = model.predict(X, verbose=0)
    return np.mean(np.square(X - X_pred), axis=1)
