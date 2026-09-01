"""
sensor_status_agent.py — Identifies which sensors deviate most from normal.

NOT WIRED INTO app/main.py — confirmed unreachable from the live FastAPI app
(2026-08, OpenAI->Qwen migration audit). Only imported by the dead
agents/orchestrator_agent.py. The live LangGraph (agents/copilot_graph.py) has
its own inline sensor_status_node instead. Left as-is.
"""
from typing import Dict, Any
from config.settings import SENSOR_SCHEMA

class SensorStatusAgent:
    def analyze_sensor(self, sensor_readings: Dict[str, float]) -> Dict[str, Any]:
        """Evaluates given sensor readings against their normal operating ranges."""
        deviations: Dict[str, Any] = {}
        suspect_sensor = None
        max_deviation = 0.0

        for sensor, value in sensor_readings.items():
            if sensor in SENSOR_SCHEMA:
                normal_min, normal_max = SENSOR_SCHEMA[sensor]["normal_range"]
                
                # Calculate deviation percentage outside normal range
                val_f = float(value)
                min_f = float(normal_min)
                max_f = float(normal_max)
                
                dev = 0.0
                if val_f < min_f:
                    dev = float((min_f - val_f) / min_f)
                elif val_f > max_f:
                    dev = float((val_f - max_f) / max_f)

                deviations[sensor] = {
                    "value": val_f,
                    "status": "normal" if dev == 0.0 else "abnormal",
                    "deviation_pct": round(float(dev * 100.0), 2)
                }

                if dev > max_deviation:
                    max_deviation = dev
                    suspect_sensor = sensor

        return {
            "deviations": deviations,
            "suspect_sensor": suspect_sensor,
            "max_deviation_pct": round(float(max_deviation * 100.0), 2)
        }
