"""
anomaly_injector.py — State-controlled anomaly injection using realistic sensor ranges.

Reads full sensor config (including min_normal, max_normal, fault thresholds)
from sensor_configs.json for newly registered machines, and falls back to
hard-coded profiles for legacy machines.
"""

import numpy as np
import os
import json


def get_machine_config(machine_id: str) -> dict:
    """
    Returns a dict of {sensor_id: (mu, sigma, min_normal, max_normal, fault_high, fault_low)}.
    Loads from DB via SensorConfigLoader; falls back to built-in profiles.
    """
    from services.sensor_config_loader import sensor_config_loader
    config = sensor_config_loader.get_machine_config(machine_id)
    
    if config:
        dyn_cfg = {}
        for sid, data in config.items():
            mu         = float(data.get("mu", 50.0))
            sigma      = float(data.get("sigma", 5.0))
            min_normal = float(data.get("min_normal", mu - 3 * sigma))
            max_normal = float(data.get("max_normal", mu + 3 * sigma))
            fault_high = float(data.get("fault_high", max_normal * 1.5))
            fl_raw     = data.get("fault_low")
            fault_low  = float(fl_raw) if fl_raw is not None else None
            dyn_cfg[sid] = (mu, sigma, min_normal, max_normal, fault_high, fault_low)
        if dyn_cfg:
            return dyn_cfg

    # ── Fallback built-in profiles ─────────────────────────────────────────
    if "LATHE" in machine_id.upper():
        return {
            # (mu, sigma, min_normal, max_normal, fault_high, fault_low)
            "temperature":   (45,   2,    30,  55,   80,    None),
            "motor_current": (12.5, 1.5,  8,   16,   25,    None),
            "vibration":     (0.15, 0.05, 0.0, 0.3,  1.0,   None),
            "speed":         (3200, 20,   3000,3500,  4000,  2500),
            "pressure":      (8.5,  0.5,  6.0, 10.0, 13.0,  None),
        }
    elif "TURBINE" in machine_id.upper():
        return {
            "temperature":   (850,  15,   750, 950,  1100,  None),
            "motor_current": (450,  5.0,  400, 500,  600,   None),
            "vibration":     (1.2,  0.2,  0.5, 2.0,  4.0,   None),
            "speed":         (15000,50,  14000,16000, 17500, 12000),
            "pressure":      (32.0, 1.0,  28,  36,   42,    None),
        }
    else:  # Default (PUMP-001)
        return {
            "temperature":   (180,  2,    170, 190,  220,   None),
            "motor_current": (4.5,  0.5,  3.0, 6.0,  9.0,   None),
            "vibration":     (0.8,  0.1,  0.3, 1.2,  2.5,   None),
            "speed":         (160,  5,    140, 180,  210,   120),
            "pressure":      (4.5,  0.2,  3.5, 5.5,  7.0,   None),
        }


# ── Reading generators ──────────────────────────────────────────────────────

def _unpack(cfg_entry):
    """Unpack a config tuple into named fields, tolerating old 2-tuple format."""
    if len(cfg_entry) == 2:
        mu, sigma = cfg_entry
        min_n, max_n = mu - 3 * sigma, mu + 3 * sigma
        return mu, sigma, min_n, max_n, max_n * 1.5, None
    mu, sigma, min_n, max_n, fault_h, fault_l = cfg_entry
    return mu, sigma, min_n, max_n, fault_h, fault_l


def normal_reading(machine_id: str = "PUMP-001") -> dict:
    """Stable readings with small Gaussian noise — healthy machine."""
    cfg = get_machine_config(machine_id)
    result = {}
    for k, entry in cfg.items():
        mu, sigma, min_n, max_n, _, _ = _unpack(entry)
        val = np.random.normal(mu, sigma)
        # Clamp to normal range to keep data realistic
        result[k] = float(np.clip(val, min_n, max_n))
    return result


def machine_fault_reading(machine_id: str = "PUMP-001") -> dict:
    """
    Simulates a mechanical fault. Uses the extracted fault thresholds to
    produce realistic out-of-range readings, not arbitrary multipliers.
    """
    cfg = get_machine_config(machine_id)
    fault_reading = {}
    for k, entry in cfg.items():
        mu, sigma, min_n, max_n, fault_h, fault_l = _unpack(entry)
        
        # Identify sensors that INCREASE during faults (heat, vibration, current)
        is_rising_fault = any(x in k.lower() for x in
                              ["temp", "vibrat", "current", "amp", "ct", "heat", "load"])
        
        if is_rising_fault:
            # Push toward fault_high with some noise
            target = fault_h
            fault_reading[k] = float(np.random.normal(target, sigma * 2))
        elif fault_l is not None:
            # Push toward fault_low (e.g., pressure drop, speed loss)
            fault_reading[k] = float(np.random.normal(fault_l, sigma * 2))
        else:
            # Moderate deviation for other sensors
            fault_reading[k] = float(np.random.normal(mu * 1.3, sigma * 3))
    return fault_reading


def sensor_freeze_reading(frozen_values: dict | None = None) -> dict:
    """Simulates a stuck/frozen sensor — all values identical to a past snapshot."""
    if frozen_values is None:
        frozen_values = normal_reading("PUMP-001")
    return {k: v for k, v in frozen_values.items()}


def sensor_drift_reading(drift_step: float = 0.0, machine_id: str = "PUMP-001") -> dict:
    """
    Simulates a sensor gradually drifting toward its fault limit.
    The first sensor in the config is chosen as the drifting one.
    """
    base = normal_reading(machine_id)
    cfg  = get_machine_config(machine_id)
    if not base:
        return base

    drift_key = list(cfg.keys())[0]
    mu, sigma, _, max_n, fault_h, _ = _unpack(cfg[drift_key])
    # Drift adds cumulatively toward fault_high
    drift_range = fault_h - mu
    base[drift_key] = float(mu + (drift_range * min(drift_step / 20.0, 1.0))
                            + np.random.normal(0, sigma * 0.5))
    return base


def idle_reading(machine_id: str = "PUMP-001") -> dict:
    """Machine is powered off / idle — near-zero or ambient readings."""
    cfg = get_machine_config(machine_id)
    idle = {}
    for k, entry in cfg.items():
        mu, sigma, _, _, _, _ = _unpack(entry)
        if any(x in k.lower() for x in ["temp", "therm"]):
            idle[k] = float(np.random.normal(25, 1))   # ambient
        elif any(x in k.lower() for x in ["pressure"]):
            idle[k] = float(np.random.normal(mu * 0.1, sigma))
        else:
            idle[k] = 0.0
    return idle


STATE_GENERATORS = {
    "normal":        normal_reading,
    "machine_fault": machine_fault_reading,
    "sensor_freeze": sensor_freeze_reading,
    "idle":          idle_reading,
}
