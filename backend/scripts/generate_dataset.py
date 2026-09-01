"""
generate_dataset.py — AI-Enhanced mock dataset generator.

Generates a realistic sensor dataset with controlled machine states 
(normal, machine_fault, sensor_freeze, sensor_drift, idle).

Now supports AI-generated anomaly patterns from AI Automation Engineer
for more realistic fault simulation.

Output: uploaded to Cloudinary (dataset_{machine_id}), no local file written.

Usage:
    python generate_dataset.py --machine_id PUMP-001
    python generate_dataset.py --machine_id PUMP-001 --use_ai_patterns
"""

import sys
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    STATE_ROW_DISTRIBUTION,
)
from simulator.anomaly_injector import (
    normal_reading,
    machine_fault_reading,
    sensor_freeze_reading,
    sensor_drift_reading,
    idle_reading,
    get_machine_config,
)
from services import pipeline_storage


def load_ai_patterns(machine_id: str) -> dict:
    """Load AI-generated anomaly patterns if available (uploaded to Cloudinary by register_machine)."""
    try:
        patterns = pipeline_storage.download_pipeline_json(
            pipeline_storage.anomaly_patterns_public_id(machine_id)
        )
        logger.info(f"✅ Loaded AI anomaly patterns for {machine_id}")
        return patterns
    except Exception as e:
        logger.info(f"No AI anomaly patterns found for {machine_id}: {e}")
        return {}


def _dict_stages(patterns) -> list:
    """
    Keep only the stage entries that are actually objects.

    The LLM is asked for a list of {"stage", "sensor_changes", ...} objects but
    intermittently returns plain strings (a prose description instead of the
    object). Calling .get() on those raised
    AttributeError: 'str' object has no attribute 'get' and killed the whole
    dataset-generation subprocess, so registration never produced a model.
    """
    if not isinstance(patterns, list):
        return []
    return [p for p in patterns if isinstance(p, dict)]


def ai_enhanced_fault_reading(machine_id: str, ai_patterns: dict, stage: int = 1) -> dict:
    """
    Generate fault reading using AI-generated patterns.
    
    Args:
        machine_id: The machine identifier
        ai_patterns: AI-generated patterns from AIAutomationEngineerAgent
        stage: Which stage of fault progression (1-N)
    """
    cfg = get_machine_config(machine_id)
    
    # Get machine_fault patterns
    fault_patterns = ai_patterns.get("machine_fault", {})
    if not isinstance(fault_patterns, dict):
        return machine_fault_reading(machine_id)
    pattern_stages = _dict_stages(fault_patterns.get("patterns"))

    if not pattern_stages:
        # Fallback to basic fault reading
        return machine_fault_reading(machine_id)

    # Select appropriate stage
    stage_idx = min(stage - 1, len(pattern_stages) - 1)
    current_stage = pattern_stages[stage_idx]
    sensor_changes = current_stage.get("sensor_changes", {})
    if not isinstance(sensor_changes, dict):
        sensor_changes = {}
    
    # Start with normal reading
    reading = normal_reading(machine_id)
    
    # Apply AI-directed changes
    for sensor_id, changes in sensor_changes.items():
        if sensor_id in reading and isinstance(changes, dict):
            direction = changes.get("direction", "increase")
            magnitude = changes.get("magnitude_factor", 1.3)
            noise = changes.get("noise_factor", 1.5)
            
            base_val = reading[sensor_id]
            
            if direction == "increase":
                reading[sensor_id] = base_val * magnitude + np.random.normal(0, abs(base_val) * 0.1 * noise)
            elif direction == "decrease":
                reading[sensor_id] = base_val / magnitude + np.random.normal(0, abs(base_val) * 0.1 * noise)
            else:
                # erratic
                reading[sensor_id] = base_val * np.random.uniform(0.8, 1.5)
    
    # Apply correlations if defined
    correlations = fault_patterns.get("correlations", [])
    if not isinstance(correlations, list):
        correlations = []
    for corr in correlations:
        if not isinstance(corr, dict):
            continue
        primary = corr.get("primary")
        secondary = corr.get("secondary")
        relationship = corr.get("relationship", "positive")
        
        if primary in reading and secondary in reading:
            primary_deviation = reading[primary] / normal_reading(machine_id).get(primary, 1) - 1
            
            if relationship == "positive":
                reading[secondary] = reading[secondary] * (1 + primary_deviation * 0.5)
            elif relationship == "negative":
                reading[secondary] = reading[secondary] * (1 - primary_deviation * 0.3)
    
    return reading


def ai_enhanced_drift_reading(drift_step: float, machine_id: str, ai_patterns: dict) -> dict:
    """
    Generate drift reading using AI-generated patterns.
    """
    # Get drift patterns if available
    drift_patterns = ai_patterns.get("sensor_drift", {})
    if not isinstance(drift_patterns, dict):
        return sensor_drift_reading(drift_step, machine_id)
    pattern_stages = _dict_stages(drift_patterns.get("patterns"))

    if not pattern_stages:
        return sensor_drift_reading(drift_step, machine_id)
    
    base = normal_reading(machine_id)
    cfg = get_machine_config(machine_id)
    
    # Apply drift based on AI patterns
    for stage in pattern_stages:
        sensor_changes = stage.get("sensor_changes", {})
        if not isinstance(sensor_changes, dict):
            continue
        for sensor_id, changes in sensor_changes.items():
            if sensor_id in base and isinstance(changes, dict):
                direction = changes.get("direction", "increase")
                magnitude = changes.get("magnitude_factor", 1.1)
                
                drift_factor = min(drift_step / 20.0, 1.0) * (magnitude - 1.0)
                
                if direction == "increase":
                    base[sensor_id] = base[sensor_id] * (1 + drift_factor)
                else:
                    base[sensor_id] = base[sensor_id] * (1 - drift_factor)
    
    return base


def generate_dataset(
    machine_id: str = "PUMP-001", 
    total_rows: int = 20_000, 
    seed: int = 42,
    use_ai_patterns: bool = False
) -> pd.DataFrame:
    """
    Generate a realistic sensor dataset using block-based state injection.
    
    Args:
        machine_id: Machine identifier
        total_rows: Number of rows to generate
        seed: Random seed for reproducibility
        use_ai_patterns: Whether to use AI-generated anomaly patterns
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Load AI patterns if requested
    ai_patterns = {}
    if use_ai_patterns:
        ai_patterns = load_ai_patterns(machine_id)
        if ai_patterns:
            logger.info(f"🤖 Using AI-enhanced anomaly patterns for {machine_id}")

    # ── Build state sequence as contiguous blocks ──────────────────────────
    state_blocks = []
    for state, count in STATE_ROW_DISTRIBUTION.items():
        remaining = count
        while remaining > 0:
            block_size = min(random.randint(10, 120), remaining)
            state_blocks.append((state, block_size))
            remaining -= block_size

    random.shuffle(state_blocks)

    # ── Generate readings block by block ───────────────────────────────────
    rows = []
    start_time = datetime(2026, 1, 1, 6, 0, 0)   # factory shift start

    tick = 0
    drift_step: float = 0.0
    frozen_snapshot = None
    fault_stage = 1

    for state, block_size in state_blocks:
        drift_step = 0.0          # reset drift each new drift block
        frozen_snapshot = None    # reset freeze snapshot
        fault_stage = 1           # reset fault progression

        for j in range(block_size):
            timestamp = start_time + timedelta(seconds=tick)

            if state == "normal":
                reading = normal_reading(machine_id)
            elif state == "machine_fault":
                if use_ai_patterns and ai_patterns:
                    reading = ai_enhanced_fault_reading(machine_id, ai_patterns, fault_stage)
                    # Progress through fault stages
                    if j % 10 == 0:
                        fault_stage = min(fault_stage + 1, 5)
                else:
                    reading = machine_fault_reading(machine_id)
            elif state == "sensor_freeze":
                if frozen_snapshot is None:
                    frozen_snapshot = normal_reading(machine_id)
                reading = sensor_freeze_reading(frozen_snapshot)
            elif state == "sensor_drift":
                drift_step = drift_step + float(random.uniform(0.05, 0.3))
                if use_ai_patterns and ai_patterns:
                    reading = ai_enhanced_drift_reading(drift_step, machine_id, ai_patterns)
                else:
                    reading = sensor_drift_reading(drift_step, machine_id)
            elif state == "idle":  # idle
                reading = idle_reading(machine_id)

            row = {
                "timestamp":     timestamp.isoformat(),
                "machine_id":    machine_id,
                "state":         state,
            }
            # Dynamically inject features
            for k, v in reading.items():
                if k not in ["timestamp", "machine_id", "state"]:
                    row[k] = round(v, 4)
                    
            rows.append(row)
            tick += 1

    df = pd.DataFrame(rows)

    # Trim or pad to exact total_rows
    if len(df) > total_rows:
        df = df.iloc[:total_rows].copy()

    df = df.reset_index(drop=True)
    
    if use_ai_patterns and ai_patterns:
        logger.info(f"✅ Generated AI-enhanced dataset with {len(df)} rows")
    
    return df


def validate_dataset(df: pd.DataFrame, machine_id: str) -> None:
    """Print a quick statistical summary of the generated dataset."""
    print("\n" + "═" * 55)
    print(f"  Dataset Summary: {machine_id}")
    print("═" * 55)
    print(f"  Total rows   : {len(df):,}")
    print(f"  Columns      : {list(df.columns)}")
    print()

    print("  State distribution:")
    counts = df["state"].value_counts()
    for state, count in counts.items():
        pct = count / len(df) * 100
        print(f"    {state:<18s} {count:>6,}  ({pct:.1f}%)")

    print()
    print("  Sensor statistics (mean ± std):")
    sensor_cols = [c for c in df.columns if c not in ["timestamp", "machine_id", "state"]]
    for col in sensor_cols:
        m = df[col].mean()
        s = df[col].std()
        print(f"    {col:<18s}  {m:8.3f} ± {s:.3f}")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate mock sensor dataset")
    parser.add_argument("--machine_id", default="PUMP-001", help="Machine ID for config")
    parser.add_argument("--rows", type=int, default=20000, help="Total rows")
    parser.add_argument("--use_ai_patterns", action="store_true", help="Use AI-generated anomaly patterns")
    args = parser.parse_args()

    print(f"Generating mock dataset for {args.machine_id} …")
    if args.use_ai_patterns:
        print("🤖 AI-enhanced anomaly patterns enabled")

    df = generate_dataset(
        machine_id=args.machine_id,
        total_rows=args.rows,
        use_ai_patterns=args.use_ai_patterns
    )
    validate_dataset(df, args.machine_id)

    url = pipeline_storage.upload_pipeline_dataframe(df, pipeline_storage.dataset_public_id(args.machine_id))
    print(f"✓ Dataset uploaded → {url}")
