"""
model_registry.py — Shared DB-first + Cloudinary asset loading for trained
anomaly-detection models/scalers/thresholds.

Used by both the real-time inference path (models/detect_anomaly.py) and the
evaluation pipeline (models/evaluate_model.py) so the cloud-loading logic only
needs to be maintained once.
"""
import logging

log = logging.getLogger(__name__)


def load_machine_assets(machine_id: str, model_type: str = "dense") -> dict:
    """
    Load {"model", "scaler", "threshold"} for a machine/model_type.

    Neon (AnomalyThreshold, MachineAsset) is the source of truth for where
    each asset lives; models/scalers are downloaded directly from Cloudinary
    into memory (models use a momentary tempfile internally, deleted
    immediately after load — see services/pipeline_storage.py). Falls back to
    PUMP-001's assets if the requested machine has none registered.
    """
    from unified_rag.db.database import SessionLocal
    from unified_rag.db.models import AnomalyThreshold, MachineAsset
    from services import pipeline_storage

    db = SessionLocal()
    try:
        # 1. Fetch threshold from DB
        t_record = db.query(AnomalyThreshold).filter(AnomalyThreshold.machine_id == machine_id,
                                                    AnomalyThreshold.threshold_type == model_type).first()
        if not t_record:
            log.warning(f"Threshold not found in DB for {machine_id}_{model_type}, falling back to PUMP-001")
            t_record = db.query(AnomalyThreshold).filter(AnomalyThreshold.machine_id == "PUMP-001",
                                                        AnomalyThreshold.threshold_type == model_type).first()

        # 2. Fetch model/scaler URLs from DB
        asset_types = [f"model_{model_type}", "scaler"]
        asset_records = db.query(MachineAsset).filter(MachineAsset.machine_id == machine_id,
                                                    MachineAsset.asset_type.in_(asset_types)).all()
        assets_meta = {a.asset_type: a.url for a in asset_records}

        if len(assets_meta) < 2:
            log.warning(f"Assets not found in DB for {machine_id}, falling back to PUMP-001")
            asset_records = db.query(MachineAsset).filter(MachineAsset.machine_id == "PUMP-001",
                                                        MachineAsset.asset_type.in_(asset_types)).all()
            assets_meta = {a.asset_type: a.url for a in asset_records}

        model_url = assets_meta.get(f"model_{model_type}")
        scaler_url = assets_meta.get("scaler")
        if not model_url or not scaler_url:
            raise FileNotFoundError(f"No cloud-registered model/scaler found for {machine_id} or PUMP-001 fallback")

        log.info(f"Loading cloud assets for {machine_id} ({model_type}) ...")
        model = pipeline_storage.download_keras_model(model_url)
        scaler = pipeline_storage.download_pickle(scaler_url)

        if t_record:
            threshold = t_record.value
        else:
            log.warning(f"No threshold in DB for {machine_id}_{model_type}, using default 0.5")
            threshold = 0.5

        return {"model": model, "scaler": scaler, "threshold": threshold}
    finally:
        db.close()
