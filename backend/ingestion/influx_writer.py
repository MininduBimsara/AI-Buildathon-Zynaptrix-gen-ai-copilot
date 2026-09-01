"""
influx_writer.py - Clean abstraction for writing points to InfluxDB.
"""
import logging
from typing import Dict

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
except ImportError:
    pass

from config.influx_config import INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET, INFLUX_MEASUREMENT

logger = logging.getLogger(__name__)

PLACEHOLDER_TOKEN = "YOUR_INFLUXDB_TOKEN"


class InfluxWriter:
    """
    Wrapper for InfluxDB connections and writing points.

    InfluxDB is optional telemetry archival — the live dashboard is fed by the
    /api/telemetry/push WebSocket path, not by this. So a missing or rejected
    token must degrade quietly: previously every tick logged a full 401 traceback
    for each simulator, which buried the actual application logs.
    """

    def __init__(self):
        self.bucket = INFLUX_BUCKET
        self.org = INFLUX_ORG
        self.measurement = INFLUX_MEASUREMENT
        self.enabled = bool(INFLUX_TOKEN) and INFLUX_TOKEN != PLACEHOLDER_TOKEN

        if not self.enabled:
            logger.info("InfluxDB archival disabled (no INFLUX_TOKEN configured).")
            self.client = None
            self.write_api = None
            return

        self.client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

    def write_sensor_reading(self, reading: Dict[str, any], state: str = "unknown"):
        """Writes a single sensor reading dict to InfluxDB."""
        if not self.enabled:
            return
        machine_id = reading.get("machine_id", "unknown_machine")
        point = (
            Point(self.measurement)
            .tag("machine_id", machine_id)
            .tag("state", state)
        )
        
        for key, value in reading.items():
            if key in ["machine_id", "state"]:
                continue
            if isinstance(value, (int, float)):
                point.field(key, round(float(value), 3))
                
        try:
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
        except Exception as e:
            # Latch off on an auth/config failure: it will fail identically on every
            # subsequent reading, so log once instead of once per tick.
            if getattr(e, "status", None) in (401, 403):
                self.enabled = False
                logger.warning(
                    "InfluxDB rejected the token (%s) — archival disabled for this run. "
                    "Check INFLUX_TOKEN/INFLUX_ORG in .env.", e.status
                )
            else:
                raise

    def close(self):
        if self.client:
            self.client.close()
