"""
alert_service.py — Alert formatting and delivery for detected anomalies.

Currently:
  - formats anomaly alerts as structured dicts
  - logs alerts to console (with severity colour codes)
  - persists alerts to Neon PostgreSQL (AnomalyRecord)

Future extensions:
  - send Slack / email webhook
  - push to MQTT topic
"""

import json
import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# ANSI colour codes for terminal output
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_RESET  = "\033[0m"


def format_alert(result: dict) -> dict:
    """
    Build a structured alert dict from a detect() result.

    Args:
        result: output of AnomalyDetector.detect() + anomaly_service fields.

    Returns:
        alert dict ready for logging / delivery.
    """
    sensors  = result.get("sensors", {})
    score    = result.get("score", 0.0)
    threshold = result.get("threshold", 0.0)

    # Find which sensor deviates most from its typical range (simple heuristic)
    suspect_sensor = _find_suspect_sensor(sensors)

    severity = "HIGH" if result.get("escalated", False) else "WARNING"

    alert = {
        "severity":              severity,
        "timestamp":             result.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "machine_id":            sensors.get("machine_id", "PUMP-001"),
        "reconstruction_score":  round(score, 6),
        "threshold":             round(threshold, 6),
        "suspect_sensor":        suspect_sensor,
        "consecutive_anomalies": result.get("consecutive_anomalies", 1),
        "sensor_readings":       {
            k: (round(v, 3) if isinstance(v, (int, float)) else v)
            for k, v in sensors.items()
        },
    }
    return alert


def log_alert(alert: dict) -> None:
    """Print alert to console and save to Postgres database."""
    colour = _RED if alert["severity"] == "HIGH" else _YELLOW
    print(
        f"\n{colour}⚠ ANOMALY ALERT [{alert['severity']}]{_RESET}\n"
        f"  Time:        {alert['timestamp']}\n"
        f"  Score:       {alert['reconstruction_score']:.5f}  "
        f"(threshold={alert['threshold']:.5f})\n"
        f"  Suspect:     {alert['suspect_sensor']}\n"
        f"  Consecutive: {alert['consecutive_anomalies']}\n"
        f"  Readings:    {alert['sensor_readings']}\n"
    )

    # Persist to Neon PostgreSQL
    try:
        from unified_rag.db.database import SessionLocal
        from unified_rag.db.models import AnomalyRecord
        
        db = SessionLocal()
        try:
            record = AnomalyRecord(
                machine_id=alert["machine_id"],
                timestamp=alert["timestamp"],
                type="ANOMALY_LOG", # Categorized as a log entry
                score=float(alert["reconstruction_score"]),
                sensor_data=json.dumps(alert["sensor_readings"]),
                severity=alert["severity"],
                suspect_sensor=alert["suspect_sensor"],
                threshold=float(alert["threshold"]),
                resolved=False
            )
            db.add(record)
            db.commit()
            log.info(f"✅ Alert synchronized to Postgres for {alert['machine_id']}")
        finally:
            db.close()
    except Exception as e:
        log.error(f"❌ Failed to persist alert to DB: {e}")


def _find_suspect_sensor(sensors: dict) -> str:
    """
    Identify the most likely faulty sensor by how much it deviates
    from its documented normal mean (from config/settings.py).
    """
    try:
        from config.settings import SENSOR_SCHEMA
        max_dev, suspect = 0.0, "unknown"
        for col, meta in SENSOR_SCHEMA.items():
            val = sensors.get(col)
            if val is None:
                continue
            mid = (meta["normal_range"][0] + meta["normal_range"][1]) / 2
            span = (meta["normal_range"][1] - meta["normal_range"][0]) / 2
            dev = abs(val - mid) / max(span, 1e-6)
            if dev > max_dev:
                max_dev, suspect = dev, col
        return suspect
    except Exception:
        return "unknown"
