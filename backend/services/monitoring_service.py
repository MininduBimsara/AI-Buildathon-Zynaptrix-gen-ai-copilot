"""
monitoring_service.py — Continuous monitoring loop linking inference with multi-agent response.

NOT WIRED INTO app/main.py — confirmed unreachable from the live FastAPI app
(2026-08, OpenAI->Qwen migration audit). Only self-referenced via its own
`if __name__ == "__main__"` manual test runner; imports the dead
agents/orchestrator_agent.py (which itself fails to import). Left as-is.

Usage (Manual testing):
    python -m services.monitoring_service
"""
import logging
import time
from typing import Dict, Any, List

from models.detect_anomaly import AnomalyDetector
from services.anomaly_service import AnomalyService
from agents.orchestrator_agent import OrchestratorAgent
from config.settings import SENSOR_COLUMNS

logger = logging.getLogger(__name__)

class MonitoringService:
    def __init__(self):
        logger.info("Initializing Monitoring Service...")
        self.detector = AnomalyDetector()
        self.orchestrator = OrchestratorAgent()
        
        # AnomalyService tracks consecutive anomalies
        self.anomaly_tracker = AnomalyService(
            consecutive_threshold=3,
            on_anomaly=self._on_anomaly_confirmed
        )
        
    def _on_anomaly_confirmed(self, alert: Dict[str, Any]):
        """Callback when AnomalyService confirms a sustained anomaly."""
        state = "machine_fault" if alert.get("severity") == "HIGH" else "machine_warning"
        score = alert.get("reconstruction_score", 0.0)
        
        logger.warning(f"🚨 Confirmed Anomaly! State: {state} (Score: {score:.2f})")
        
        full_alert_data = {
            "machine_state": state,
            "anomaly_score": score,
            "suspect_sensor": alert.get("suspect_sensor"),
            "recent_readings": alert.get("sensor_readings", {})
        }
        
        # Trigger orchestrator
        result = self.orchestrator.handle_anomaly(full_alert_data)
        logger.info(f"✅ Orchestrator finished handling event {result.get('event_id')}")
        print("\n" + "="*60)
        print(f"🤖 ORCHESTRATOR RESPONSE (Event ID: {result.get('event_id')})")
        print("="*60)
        print(f"Machine State: {result.get('machine_state')}")
        print(f"Suspect Sensor: {result.get('suspect_sensor')}")
        print(f"Strategy: {result.get('strategy')}\n")
        print(f"Knowledge Agent Advice:\n{result.get('rag_advice')}")
        print("="*60 + "\n")
        
    def process_reading(self, reading_dict: Dict[str, float], timestamp: str):
        """Process a single sensor reading and handle anomalies automatically."""
        # Track in anomaly service (this calls _on_anomaly_confirmed if threshold met)
        result = self.anomaly_tracker.process(reading_dict)
        return result


# =============================================================================
# CLI manual test runner
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    print("Starting manual test of Phase 4 Multi-Agent Pipeline...")
    
    # columns: temperature, motor_current, vibration, speed, pressure
    normal_reading = {"temperature": 180.0, "motor_current": 4.5, "vibration": 0.8, "speed": 160.0, "pressure": 4.5}
    fault_reading  = {"temperature": 181.0, "motor_current": 7.5, "vibration": 2.8, "speed": 120.0, "pressure": 4.2} # High motor current, high vibration
    
    svc = MonitoringService()
    
    print("\n[Simulator] Sending Normal Readings...")
    for i in range(3):
        print(f"  -> Normal Reading {i+1}")
        svc.process_reading(normal_reading, f"T{i}")
        time.sleep(0.5)
        
    print("\n[Simulator] Sending Fault Readings...")
    for i in range(3):
        print(f"  -> Fault Reading {i+1}")
        svc.process_reading(fault_reading, f"F{i}")
        time.sleep(0.5)
        
    print("\nManual test complete. 🚀")
