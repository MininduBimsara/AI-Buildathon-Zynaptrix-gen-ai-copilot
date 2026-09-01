"""
pipeline_storage.py — Cloud round-trip helpers for pipeline data hand-offs.

The single place that knows how to move data to/from Cloudinary without ever
touching backend/data/. Used for the training pipeline's inter-stage hand-offs
(dataset -> normalized dataset -> model/scaler) and evaluation artifacts.

Everything is in-memory except Keras models, which TensorFlow's save/load API
requires a real filesystem path for -- that case uses a NamedTemporaryFile
that's always deleted in a finally block, never written into backend/data/.

Pipeline stages run as separate subprocesses (scripts/generate_dataset.py ->
preprocessing/normalization.py -> models/train_model.py -> models/evaluate_model.py),
so a stage can't just hand its output URL to the next stage in memory. Instead
each stage uploads under a deterministic public_id (dataset_{machine_id},
normalized_{machine_id}, ...) and the next stage looks up the CURRENT url via
Cloudinary's Admin API (resolve_url) rather than reconstructing a delivery URL
by hand -- that avoids any CDN staleness after an overwrite= upload.
"""
import io
import json
import pickle
import tempfile
import os

import pandas as pd
import requests
import cloudinary.api

from services.cloudinary_service import CloudinaryService

cloudinary_service = CloudinaryService()

PIPELINE_FOLDER = "industrial_copilot/pipeline"


def dataset_public_id(machine_id: str) -> str:
    return f"dataset_{machine_id}"


def normalized_public_id(machine_id: str) -> str:
    return f"normalized_{machine_id}"


def anomaly_patterns_public_id(machine_id: str) -> str:
    return f"anomaly_patterns_{machine_id}"


def resolve_url(public_id: str, folder: str = PIPELINE_FOLDER, resource_type: str = "raw") -> str:
    """Look up the current Cloudinary URL for a known public_id via the Admin API."""
    resource = cloudinary.api.resource(f"{folder}/{public_id}", resource_type=resource_type)
    return resource["secure_url"]


def upload_dataframe(df: pd.DataFrame, public_id: str, folder: str) -> str | None:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return cloudinary_service.upload_file(
        io.BytesIO(buf.getvalue().encode("utf-8")), public_id, folder, resource_type="raw"
    )


def download_dataframe(url: str) -> pd.DataFrame:
    resp = requests.get(url)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def upload_json(data: dict, public_id: str, folder: str) -> str | None:
    payload = json.dumps(data, default=str).encode("utf-8")
    return cloudinary_service.upload_file(io.BytesIO(payload), public_id, folder, resource_type="raw")


def download_json(url: str) -> dict:
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def upload_pickle(obj, public_id: str, folder: str) -> str | None:
    payload = pickle.dumps(obj)
    return cloudinary_service.upload_file(io.BytesIO(payload), public_id, folder, resource_type="raw")


def download_pickle(url: str):
    resp = requests.get(url)
    resp.raise_for_status()
    return pickle.loads(resp.content)


def upload_keras_model(model, public_id: str, folder: str) -> str | None:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            tmp_path = tmp.name
        model.save(tmp_path)
        return cloudinary_service.upload_file(tmp_path, public_id, folder, resource_type="raw")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def download_keras_model(url: str):
    import tensorflow as tf

    resp = requests.get(url)
    resp.raise_for_status()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name
        return tf.keras.models.load_model(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── Pipeline-scoped convenience wrappers ─────────────────────────────────────
# (upload/download by public_id alone, resolving the current URL via the Admin
# API -- lets each pipeline stage stay self-sufficient given just machine_id)

def upload_pipeline_dataframe(df: pd.DataFrame, public_id: str) -> str | None:
    return upload_dataframe(df, public_id, PIPELINE_FOLDER)


def download_pipeline_dataframe(public_id: str) -> pd.DataFrame:
    return download_dataframe(resolve_url(public_id))


def upload_pipeline_json(data: dict, public_id: str) -> str | None:
    return upload_json(data, public_id, PIPELINE_FOLDER)


def download_pipeline_json(public_id: str) -> dict:
    return download_json(resolve_url(public_id))
