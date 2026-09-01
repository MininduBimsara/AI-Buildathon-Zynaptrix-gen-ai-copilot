"""
normalization.py — Sensor data normalization utilities.

Uses StandardScaler fitted only on NORMAL operating state data
so anomalous readings remain detectable (not normalized away).

Reads the raw dataset from Cloudinary (uploaded by scripts/generate_dataset.py)
and uploads the normalized dataset + scaler back to Cloudinary/Neon — no local
files are read or written.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.preprocessing import StandardScaler

from services import pipeline_storage


def fit_scaler(df: pd.DataFrame) -> StandardScaler:
    """
    Fit a StandardScaler on normal-state data only.
    """
    # Detect sensor columns (all numeric columns except metadata)
    sensor_cols = [c for c in df.columns if c not in ["timestamp", "machine_id", "state"]]
    
    if "state" in df.columns:
        fit_df = df[df["state"] == "normal"][sensor_cols]
    else:
        fit_df = df[sensor_cols]

    scaler = StandardScaler()
    scaler.fit(fit_df)
    return scaler

def save_scaler(scaler: StandardScaler, machine_id: str = "PUMP-001") -> None:
    """Upload the scaler directly to Cloudinary (in-memory pickle, no local file) and register it in Neon."""
    try:
        from unified_rag.db.database import SessionLocal
        from unified_rag.db.models import MachineAsset
        from datetime import datetime

        print(f"☁️ Uploading scaler for {machine_id} to Cloudinary...")
        url = pipeline_storage.upload_pickle(
            scaler, public_id=f"scaler_{machine_id}", folder="industrial_copilot/assets/scalers"
        )

        if url:
            db = SessionLocal()
            try:
                asset = db.query(MachineAsset).filter(MachineAsset.machine_id == machine_id,
                                                     MachineAsset.asset_type == "scaler").first()
                if not asset:
                    asset = MachineAsset(machine_id=machine_id, asset_type="scaler")
                    db.add(asset)

                asset.url = url
                asset.updated_at = datetime.now().isoformat()
                db.commit()
                print(f"✅ Scaler registered in DB: {url}")
            finally:
                db.close()
    except Exception as e:
        print(f"⚠️ Cloud scaler sync failed: {e}")

def load_scaler(machine_id: str = "PUMP-001") -> StandardScaler:
    """Load the scaler from its registered Cloudinary URL (Neon is the source of truth for the URL)."""
    from unified_rag.db.database import SessionLocal
    from unified_rag.db.models import MachineAsset

    db = SessionLocal()
    try:
        asset = db.query(MachineAsset).filter(MachineAsset.machine_id == machine_id,
                                             MachineAsset.asset_type == "scaler").first()
        if not asset:
            raise FileNotFoundError(f"No scaler registered for {machine_id}")
        return pipeline_storage.download_pickle(asset.url)
    finally:
        db.close()

def normalize(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Return a copy of df with sensor columns standardized."""
    df = df.copy()
    sensor_cols = [c for c in df.columns if c not in ["timestamp", "machine_id", "state"]]
    df[sensor_cols] = scaler.transform(df[sensor_cols])
    return df

def fit_and_normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Convenience: fit scaler then normalize."""
    scaler = fit_scaler(df)
    return normalize(df, scaler), scaler

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Normalize sensor data")
    parser.add_argument("--machine_id", default="PUMP-001", help="Machine ID")
    args = parser.parse_args()

    df = pipeline_storage.download_pipeline_dataframe(pipeline_storage.dataset_public_id(args.machine_id))
    normalized_df, scaler = fit_and_normalize(df)

    url = pipeline_storage.upload_pipeline_dataframe(
        normalized_df, pipeline_storage.normalized_public_id(args.machine_id)
    )
    save_scaler(scaler, args.machine_id)

    print(f"✓ Normalized data uploaded → {url}")
