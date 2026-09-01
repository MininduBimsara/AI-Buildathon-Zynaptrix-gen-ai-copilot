"""
sensor_simulator.py — Real-time sensor data simulator.

Streams synthetic sensor readings every second into InfluxDB.
Uses the same anomaly_injector state machine as the batch dataset generator
so the streaming data distribution matches training data.

Usage:
    python sensor_simulator.py

Environment variables required (.env):
    INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET
"""

import sys
import time
import random
import logging
import requests
from datetime import datetime, timezone

# Allow running from any working directory
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.anomaly_injector import (
    normal_reading,
    machine_fault_reading,
    sensor_freeze_reading,
    sensor_drift_reading,
    idle_reading,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# Removed old get_influx_writer function
from unified_rag.config import settings

# ─────────────────────────────────────────────────────────────────────────────
# 🔧 Config & Consts
# ─────────────────────────────────────────────────────────────────────────────
# Root API URL for pushing telemetry
API_BASE_URL = settings.api_url
TELEMETRY_ENDPOINT = f"{API_BASE_URL}/api/telemetry/push"


def pick_state(current_state: str, drift_counter: list) -> str:
    """
    State machine transition.
    Once in a fault/freeze/drift state, persist for a random burst
    to simulate realistic event duration.
    """
    r = random.random()
    if r < 0.70:
        return "normal"
    elif r < 0.85:
        return "machine_fault"
    elif r < 0.93:
        return "sensor_freeze"
    elif r < 0.97:
        return "sensor_drift"
    else:
        return "idle"


def simulate(machine_id: str = "PUMP-001", interval_seconds: float = 1.0):
    """Main simulation loop — streams sensor data every interval_seconds."""
    from ingestion.influx_writer import InfluxWriter
    writer = InfluxWriter()

    logger.info(f"▶ Sensor simulator started for {machine_id}. Press Ctrl+C to stop.")

    state = "normal"
    drift_step = 0.0
    frozen_snapshot = None
    state_duration = 0       # how many ticks remain in current state
    state_counter = 0        # ticks elapsed in current state

    try:
        while True:
            # Transition state
            if state_counter >= state_duration:
                state = pick_state(state, [])
                state_duration = random.randint(3, 30)   # persist state for 3–30 ticks
                state_counter = 0
                drift_step = 0.0
                frozen_snapshot = None
                logger.info(f"  [{machine_id}] → State changed to: {state} (duration: {state_duration}s)")

            # Generate reading
            if state == "normal":
                reading = normal_reading(machine_id=machine_id)
            elif state == "machine_fault":
                reading = machine_fault_reading(machine_id=machine_id)
            elif state == "sensor_freeze":
                if frozen_snapshot is None:
                    frozen_snapshot = normal_reading(machine_id=machine_id)   # freeze at a normal snapshot
                reading = sensor_freeze_reading(frozen_snapshot)
            elif state == "sensor_drift":
                drift_step += random.uniform(0.1, 0.5)
                reading = sensor_drift_reading(drift_step, machine_id=machine_id)
            else:
                reading = idle_reading(machine_id=machine_id)

            # Add machine context
            reading["machine_id"] = machine_id

            # Write to InfluxDB with absolute safety
            try:
                writer.write_sensor_reading(reading, state=state)
            except Exception as e:
                logger.warning(f"  [{machine_id}] ⚠️ InfluxDB Write Failed (Continuing): {e}")
            
            # Broadcast to UI WebSockets
            try:
                # 2s was shorter than a push takes when the server is also running model
                # inference and a manual ingestion, so readings were dropped
                # client-side even though the server answered 200.
                resp = requests.post(TELEMETRY_ENDPOINT, json=reading, timeout=15)
                if resp.status_code != 200:
                    logger.warning(f"  [{machine_id}] ⚠️ Telemetry push failed (Status {resp.status_code})")
            except Exception as ex:
                logger.error(f"  [{machine_id}] ❌ Telemetry push failed: {ex}")

            # Build dynamic log string from reading
            vals = [f"{k}={v:.2f}" for k, v in reading.items() if isinstance(v, (int, float))]
            log_str = "  ".join(vals[:4]) + ("..." if len(vals) > 4 else "")
            
            logger.info(
                f"  [{machine_id:10s}] [{state:15s}] {log_str}"
            )

            state_counter += 1
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        logger.info(f"■ Simulator stopped for {machine_id}.")
    finally:
        writer.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Industrial Sensor Simulator")
    parser.add_argument("--machine_id", type=str, default="PUMP-001", help="ID of the machine being simulated")
    args = parser.parse_args()
    
    simulate(machine_id=args.machine_id)
