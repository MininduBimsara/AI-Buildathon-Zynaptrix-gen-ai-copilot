"""Download real machine models and scalers from Cloudinary."""
import requests
import os

ASSETS = {
    "PUMP-001": {
        "model": "https://res.cloudinary.com/dp3pidybx/raw/upload/v1776030369/industrial_copilot/assets/model_dense/asset_model_dense_PUMP-001.keras",
        "scaler": "https://res.cloudinary.com/dp3pidybx/raw/upload/v1776030374/industrial_copilot/assets/scaler/asset_scaler_PUMP-001.pkl",
    },
    "LATHE-002": {
        "model": "https://res.cloudinary.com/dp3pidybx/raw/upload/v1776030368/industrial_copilot/assets/model_dense/asset_model_dense_LATHE-002.keras",
        "scaler": "https://res.cloudinary.com/dp3pidybx/raw/upload/v1776030373/industrial_copilot/assets/scaler/asset_scaler_LATHE-002.pkl",
    },
    "TURBINE-003": {
        "model": "https://res.cloudinary.com/dp3pidybx/raw/upload/v1776030371/industrial_copilot/assets/model_dense/asset_model_dense_TURBINE-003.keras",
        "scaler": "https://res.cloudinary.com/dp3pidybx/raw/upload/v1776030376/industrial_copilot/assets/scaler/asset_scaler_TURBINE-003.pkl",
    },
}

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

for machine_id, urls in ASSETS.items():
    for asset_type, url in urls.items():
        ext = ".keras" if asset_type == "model" else ".pkl"
        prefix = "autoencoder" if asset_type == "model" else "scaler"
        fname = f"{prefix}_{machine_id}{ext}"
        path = os.path.join(OUT_DIR, fname)
        if os.path.exists(path):
            print(f"  Already exists: {fname}")
            continue
        print(f"  Downloading {fname} ...")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"  Saved {fname}  ({len(r.content) // 1024} KB)")

print("\nDone. Files in", OUT_DIR)
for f in os.listdir(OUT_DIR):
    if f.endswith((".keras", ".pkl")):
        sz = os.path.getsize(os.path.join(OUT_DIR, f)) // 1024
        print(f"  {f}  ({sz} KB)")
