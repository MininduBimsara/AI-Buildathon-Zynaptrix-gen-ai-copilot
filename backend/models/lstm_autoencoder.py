"""
lstm_autoencoder.py — LSTM-based Autoencoder for time-series anomaly detection.

Encodes a window of T timesteps and reconstructs it, capturing temporal
dependencies that a Dense autoencoder misses.

Input shape: (batch, timesteps, n_features)

Architecture:
    LSTM(32)  → encoder
    RepeatVector(T)
    LSTM(32, return_sequences=True)  → decoder
    TimeDistributed(Dense(n_features))  → reconstruction
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Dense, RepeatVector,
    TimeDistributed, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def build_lstm_autoencoder(timesteps: int = 30,
                            n_features: int = 5,
                            latent_units: int = 16,
                            dropout_rate: float = 0.1) -> Model:
    """
    Build and compile an LSTM Autoencoder.

    Args:
        timesteps:    Window length (number of time steps per sample).
        n_features:   Number of sensors per time step.
        latent_units: Hidden units in encoder/decoder LSTM layers.
        dropout_rate: Recurrent dropout for regularization.

    Returns:
        Compiled Keras Model.
    """
    inputs = Input(shape=(timesteps, n_features), name="lstm_input")

    # ── Encoder ──────────────────────────────────
    x = LSTM(latent_units * 2, activation="tanh",
             return_sequences=False,
             recurrent_dropout=dropout_rate,
             name="enc_lstm")(inputs)

    # ── Bottleneck broadcast ──────────────────────
    x = RepeatVector(timesteps, name="bottleneck")(x)

    # ── Decoder ──────────────────────────────────
    x = LSTM(latent_units * 2, activation="tanh",
             return_sequences=True,
             recurrent_dropout=dropout_rate,
             name="dec_lstm")(x)

    outputs = TimeDistributed(
        Dense(n_features, activation="linear"), name="reconstruction"
    )(x)

    model = Model(inputs, outputs, name="LSTMAutoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse",
                  metrics=["mae"])
    return model


def create_sequences(data: np.ndarray, timesteps: int = 30) -> np.ndarray:
    """
    Slide a window of `timesteps` over the data array.

    Args:
        data:       2-D array of shape (n_samples, n_features).
        timesteps:  Window size.

    Returns:
        3-D array of shape (n_windows, timesteps, n_features).
    """
    sequences = []
    for i in range(len(data) - timesteps):
        sequences.append(data[i : i + timesteps])
    return np.array(sequences)


def reconstruction_error(model: Model, X_seq: np.ndarray) -> np.ndarray:
    """
    Compute per-sequence MSE (averaged over timesteps and features).

    Returns:
        1-D array of shape (n_sequences,) — one anomaly score per window.
    """
    X_pred = model.predict(X_seq, verbose=0)
    return np.mean(np.square(X_seq - X_pred), axis=(1, 2))


def get_callbacks(patience: int = 10) -> list:
    return [
        EarlyStopping(monitor="val_loss", patience=patience,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1),
    ]
