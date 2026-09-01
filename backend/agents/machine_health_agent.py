"""
machine_health_agent.py — Queries InfluxDB to compute a summary of current machine health.

NOT WIRED INTO app/main.py — confirmed unreachable from the live FastAPI app
(2026-08, OpenAI->Qwen migration audit). Only imported by the dead
agents/orchestrator_agent.py. Left as-is.
"""
import logging
from typing import Dict, Any

from influxdb_client import InfluxDBClient
from config.influx_config import INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET, INFLUX_MEASUREMENT

logger = logging.getLogger(__name__)

class MachineHealthAgent:
    def __init__(self):
        self.client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        self.query_api = self.client.query_api()

    def check_health(self, minutes_back: int = 5) -> Dict[str, Any]:
        """Queries the last N minutes of sensor data and returns a health summary."""
        query = f'''
            from(bucket:"{INFLUX_BUCKET}")
            |> range(start: -{minutes_back}m)
            |> filter(fn: (r) => r._measurement == "{INFLUX_MEASUREMENT}")
            |> mean()
        '''
        try:
            tables = self.query_api.query(query)
            sensor_means = {}
            for table in tables:
                for record in table.records:
                    # InfluxDB stores the field name in _field and average in _value
                    sensor_means[record.get_field()] = round(record.get_value(), 2)
            
            return {
                "status": "online" if sensor_means else "offline/no data",
                "recent_averages": sensor_means,
                "time_window": f"last {minutes_back} minutes"
            }
        except Exception as e:
            logger.error(f"[HealthAgent] Failed to query InfluxDB: {e}")
            return {"error": str(e), "status": "unknown"}
    
    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
