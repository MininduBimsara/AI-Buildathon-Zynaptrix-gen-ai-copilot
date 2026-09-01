import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from unified_rag.ai_client import get_client, MODEL_CHAT
from unified_rag.db.database import SessionLocal
from unified_rag.db.models import AnomalyRecord

from services.sensor_config_loader import sensor_config_loader
from services.anomaly_service import AnomalyService, TemporalAnalyzer, calculate_hybrid_confidence
from agents.validation_prompts import build_validation_prompt, get_full_system_prompt
from app.websockets import broadcast_anomaly

logger = logging.getLogger(__name__)

async def run_ai_validation(record_id: int, machine_id: str, anomaly_score: float, sensor_readings: dict):
    """
    Asynchronous AI validation layer using LLM to verify detected anomalies.
    """
    db = SessionLocal()
    try:
        record = db.query(AnomalyRecord).filter(AnomalyRecord.id == record_id).first()
        if not record:
            logger.error(f"Anomaly record {record_id} not found for AI validation")
            return

        # 1. Physics Validation
        physics_summary = sensor_config_loader.get_violation_summary(machine_id, sensor_readings)
        
        # 2. Temporal Analysis (Simplified for real-time)
        # In a production system, we'd fetch historical readings from DB/Influx here
        temporal_pattern = {
            "is_spike": False,
            "is_sustained": True,
            "anomaly_count": 1,
            "trend": "stable",
            "max_rate_of_change": 0.0,
            "suspicious_sensors": []
        }
        
        # 3. Hybrid Confidence Calculation
        hybrid_confidence = calculate_hybrid_confidence(
            anomaly_score, physics_summary, temporal_pattern
        )
        
        # 4. Auto-classification for obvious low-confidence cases
        if hybrid_confidence < 0.2:
            record.ai_validation_status = "SENSOR_GLITCH"
            record.fault_category = None
            record.ai_confidence_score = 0.85
            record.ai_engineering_notes = "Low hybrid confidence indicates transient noise. No action required."
            db.commit()
            logger.info(f"⚡ [Validation] Auto-classified anomaly {record.id} as SENSOR_GLITCH")
        else:
            # 5. LLM-based Intelligent Validation
            ai_client = get_client()

            prompt = build_validation_prompt(
                machine_id=machine_id,
                ml_score=anomaly_score,
                physics_summary=physics_summary,
                temporal_pattern=temporal_pattern,
                recent_readings=sensor_readings,
                hybrid_confidence=hybrid_confidence
            )
            
            try:
                response = ai_client.chat.completions.create(
                    model=MODEL_CHAT,
                    messages=[
                        {"role": "system", "content": get_full_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )
                
                validation_json = json.loads(response.choices[0].message.content)
                
                record.ai_validation_status = validation_json.get("ai_validation_status", "TRUE_FAULT")
                record.fault_category = validation_json.get("fault_category")
                record.ai_confidence_score = validation_json.get("confidence_score", hybrid_confidence)
                record.ai_engineering_notes = validation_json.get("ai_engineering_notes", "Validation completed.")
                db.commit()
                
                logger.info(f"✅ [Validation] Anomaly {record.id}: {record.ai_validation_status} | {record.fault_category or 'N/A'} | Confidence: {record.ai_confidence_score:.2f}")
            except Exception as llm_err:
                logger.error(f"LLM validation failed for anomaly {record.id}: {llm_err}")
                record.ai_validation_status = "TRUE_FAULT"
                record.fault_category = "mechanical"
                record.ai_confidence_score = hybrid_confidence
                record.ai_engineering_notes = "AI validation pending - LLM error encountered."
                db.commit()

        # 6. Broadcast updated record with AI validation info
        full_alert_data = {
            "id": record.id,
            "machine_id": machine_id,
            "machine_state": record.type,
            "anomaly_score": anomaly_score,
            "recent_readings": sensor_readings,
            "ai_validation_status": record.ai_validation_status,
            "fault_category": record.fault_category,
            "ai_confidence_score": record.ai_confidence_score,
            "ai_engineering_notes": record.ai_engineering_notes
        }
        
        await broadcast_anomaly(full_alert_data)

    except Exception as e:
        logger.error(f"Failed AI validation for record {record_id}: {e}")
    finally:
        db.close()

def on_anomaly_discovered(alert: dict):
    """
    Main entry point for handling a discovered anomaly.
    Persists to DB and triggers async AI validation.
    """
    state = "machine_fault" if alert.get("severity") == "HIGH" else "machine_warning"
    machine_id = alert.get("machine_id", "PUMP-001")
    sensor_readings = alert.get("sensor_readings", {})
    anomaly_score = alert.get("reconstruction_score", 0.1)

    db = SessionLocal()
    try:
        record = AnomalyRecord(
            machine_id=machine_id,
            timestamp=alert.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            type=state,
            score=int(anomaly_score * 100),
            sensor_data=json.dumps(sensor_readings)
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        
        # Schedule AI Validation in background
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(run_ai_validation(record.id, machine_id, anomaly_score, sensor_readings))
        except RuntimeError:
            # For environments without a running loop
            asyncio.run(run_ai_validation(record.id, machine_id, anomaly_score, sensor_readings))

    except Exception as e:
        logger.error(f"Failed to persist anomaly: {e}")
    finally:
        db.close()

# Registry of stateful trackers (isolates consecutive counts per machine)
anomaly_trackers: Dict[str, AnomalyService] = {}

def get_tracker(machine_id: str) -> AnomalyService:
    if machine_id not in anomaly_trackers:
        anomaly_trackers[machine_id] = AnomalyService(
            consecutive_threshold=3,
            on_anomaly=on_anomaly_discovered
        )
    return anomaly_trackers[machine_id]
