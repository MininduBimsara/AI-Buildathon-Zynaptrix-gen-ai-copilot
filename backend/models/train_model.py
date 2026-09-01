"""
train_model.py — Training pipeline for the Dense and LSTM Autoencoders.

Training rules:
  - ONLY normal-state rows are used for training
  - Validation split is taken from normal rows
  - Anomaly threshold is computed as:  mean + 2*std  of training reconstruction errors
  - Trained model is uploaded directly to Cloudinary and registered in Neon
    (MachineAsset); the threshold is written straight to Neon (AnomalyThreshold).
    No local files are read or written.

Usage:
    python models/train_model.py --model dense   (default)
    python models/train_model.py --model lstm
"""

import sys
import os
import argparse
import logging

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.autoencoder_model import (
    build_autoencoder,
    get_callbacks as dense_callbacks,
    reconstruction_error as dense_error,
)
from models.lstm_autoencoder import (
    build_lstm_autoencoder,
    create_sequences,
    get_callbacks as lstm_callbacks,
    reconstruction_error as lstm_error,
)
from services import pipeline_storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TIMESTEPS = 30   # window length for LSTM


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_normal_data(machine_id: str = "PUMP-001") -> tuple[np.ndarray, int]:
    """Load the normalized dataset (uploaded by preprocessing/normalization.py) and return only normal-state rows."""
    df = pipeline_storage.download_pipeline_dataframe(pipeline_storage.normalized_public_id(machine_id))
    sensor_cols = [c for c in df.columns if c not in ["timestamp", "machine_id", "state"]]
    normal_df = df[df["state"] == "normal"][sensor_cols]
    log.info(f"[{machine_id}] Normal rows for training: {len(normal_df):,} with {len(sensor_cols)} sensors")
    return normal_df.values.astype(np.float32), len(sensor_cols)


def compute_threshold(errors: np.ndarray, n_sigma: float = 2.0) -> float:
    """Threshold = mean + n_sigma * std of training reconstruction errors."""
    threshold = float(np.mean(errors) + n_sigma * np.std(errors))
    log.info(f"Computed threshold: {threshold:.6f}  (mean={np.mean(errors):.6f}, std={np.std(errors):.6f})")
    return threshold


def save_threshold(machine_id: str, key: str, value: float) -> None:
    # Save to Postgres (DB is the single source of truth)
    try:
        from unified_rag.db.database import SessionLocal
        from unified_rag.db.models import AnomalyThreshold
        from datetime import datetime

        db = SessionLocal()
        try:
            record = db.query(AnomalyThreshold).filter(AnomalyThreshold.machine_id == machine_id,
                                                        AnomalyThreshold.threshold_type == key).first()
            if not record:
                record = AnomalyThreshold(machine_id=machine_id, threshold_type=key)
                db.add(record)

            record.value = float(value)
            record.updated_at = datetime.now().isoformat()
            db.commit()
            log.info(f"✅ Threshold synchronized to DB for {machine_id}_{key}")
        finally:
            db.close()
    except Exception as e:
        log.error(f"⚠️ Threshold DB sync failed: {e}")


def upload_model_to_cloud(model, machine_id: str, model_type: str):
    try:
        from unified_rag.db.database import SessionLocal
        from unified_rag.db.models import MachineAsset
        from datetime import datetime

        log.info(f"☁️ Uploading {model_type} model for {machine_id} to Cloudinary...")
        url = pipeline_storage.upload_keras_model(
            model, public_id=f"model_{model_type}_{machine_id}", folder="industrial_copilot/assets/models"
        )
        if url:
            db = SessionLocal()
            try:
                asset = db.query(MachineAsset).filter(MachineAsset.machine_id == machine_id,
                                                     MachineAsset.asset_type == f"model_{model_type}").first()
                if not asset:
                    asset = MachineAsset(machine_id=machine_id, asset_type=f"model_{model_type}")
                    db.add(asset)
                asset.url = url
                asset.updated_at = datetime.now().isoformat()
                db.commit()
                log.info(f"✅ Model registered in DB: {url}")
            finally:
                db.close()
    except Exception as e:
        log.error(f"⚠️ Cloud model sync failed: {e}")


# ── Training routines ─────────────────────────────────────────────────────────

def train_dense(machine_id: str = "PUMP-001"):
    log.info("═" * 50)
    log.info(f"Training Dense Autoencoder for {machine_id}")
    log.info("═" * 50)

    X, n_features = load_normal_data(machine_id)
    split = int(len(X) * 0.85)
    X_train, X_val = X[:split], X[split:]
    log.info(f"Train: {len(X_train):,}   Val: {len(X_val):,}")

    model = build_autoencoder(n_features=n_features)
    model.summary()

    model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=100,
        batch_size=64,
        callbacks=dense_callbacks(patience=12),
        verbose=2,
    )

    upload_model_to_cloud(model, machine_id, "dense")

    # Compute and save threshold
    errors = dense_error(model, X_train)
    threshold = compute_threshold(errors)
    save_threshold(machine_id, "dense", threshold)

    log.info("Dense autoencoder training complete ✓")
    return model, threshold


def train_lstm(machine_id: str = "PUMP-001"):
    log.info("═" * 50)
    log.info(f"Training LSTM Autoencoder for {machine_id}")
    log.info("═" * 50)

    X, n_features = load_normal_data(machine_id)
    X_seq = create_sequences(X, timesteps=TIMESTEPS)
    log.info(f"Sequences shape: {X_seq.shape}")

    split = int(len(X_seq) * 0.85)
    X_train, X_val = X_seq[:split], X_seq[split:]

    model = build_lstm_autoencoder(timesteps=TIMESTEPS, n_features=n_features)
    model.summary()

    model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=60,
        batch_size=64,
        callbacks=lstm_callbacks(patience=8),
        verbose=2,
    )

    upload_model_to_cloud(model, machine_id, "lstm")

    errors = lstm_error(model, X_train)
    threshold = compute_threshold(errors)
    save_threshold(machine_id, "lstm", threshold)

    log.info("LSTM autoencoder training complete ✓")
    return model, threshold


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-machine Model Training")
    parser.add_argument("--model", choices=["dense", "lstm"], default="dense")
    parser.add_argument("--machine_id", default="PUMP-001", help="Machine to train for")
    args = parser.parse_args()

    if args.model == "dense":
        train_dense(args.machine_id)
    else:
        train_lstm(args.machine_id)
