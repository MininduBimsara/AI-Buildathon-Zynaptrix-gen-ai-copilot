"""
settings.py — Central configuration for all sensor definitions,
machine states, and dataset generation parameters.
"""

# ─────────────────────────────────────────────
# Sensor Schema
# ─────────────────────────────────────────────
SENSOR_SCHEMA = {
    "temperature": {
        "description": "Sealing heater temperature",
        "unit": "°C",
        "normal_range": (175, 185),
        "normal_mean": 180,
        "normal_std": 2,
    },
    "motor_current": {
        "description": "Motor electrical load",
        "unit": "A",
        "normal_range": (3, 6),
        "normal_mean": 4.5,
        "normal_std": 0.5,
    },
    "vibration": {
        "description": "Mechanical vibration",
        "unit": "mm/s",
        "normal_range": (0.5, 1.2),
        "normal_mean": 0.8,
        "normal_std": 0.1,
    },
    "speed": {
        "description": "Production speed (bags per minute)",
        "unit": "bpm",
        "normal_range": (150, 170),
        "normal_mean": 160,
        "normal_std": 5,
    },
    "pressure": {
        "description": "Pneumatic pressure",
        "unit": "bar",
        "normal_range": (4.0, 5.0),
        "normal_mean": 4.5,
        "normal_std": 0.2,
    },
}

SENSOR_COLUMNS = list(SENSOR_SCHEMA.keys())

# ─────────────────────────────────────────────
# Machine State Definitions
# ─────────────────────────────────────────────
MACHINE_STATES = {
    "normal": {
        "description": "Machine operating normally",
        "probability": 0.70,
    },
    "machine_fault": {
        "description": "Mechanical fault — motor wear / overload",
        "probability": 0.15,
    },
    "sensor_freeze": {
        "description": "Sensor stops updating (stuck value)",
        "probability": 0.08,
    },
    "sensor_drift": {
        "description": "Sensor gradually shifts from true value",
        "probability": 0.04,
    },
    "idle": {
        "description": "Machine stopped / powered down",
        "probability": 0.03,
    },
}

# Approximate row counts for 20,000-row dataset
DATASET_SIZE = 20_000
STATE_ROW_DISTRIBUTION = {
    "normal":        14_000,
    "machine_fault":  3_000,
    "sensor_freeze":  1_500,
    "sensor_drift":   1_000,
    "idle":             500,
}

# ─────────────────────────────────────────────
# Anomaly Thresholds (for detection layer)
# ─────────────────────────────────────────────
ANOMALY_THRESHOLD = 0.02   # MSE reconstruction error threshold
