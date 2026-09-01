"""
eval_sensitivity.py — Sensitivity analysis for IRAI26 Action Item #5

Sweeps α_temp and β_spike ±50% from baseline and measures the effect on
Precision and FPR for the TEAPM_001 autoencoder with hybrid confidence filtering.

Physics confidence formula:
    C_hybrid = ml_score
             + 0.30  (critical physics violation)    ← α_phys_fault
             + 0.15  (warning physics violation)     ← α_phys_warn
             + α_temp (sustained anomaly trend)
             - β_spike (sudden spike)
    clipped to [0, 1]

    Readings with C_hybrid < 0.2 → SENSOR_GLITCH (predicted normal)

Usage (from backend/):
    & "C:\\ProgramData\\miniconda3\\python.exe" scripts/eval_sensitivity.py

Output:
    data/processed/evaluation/sensitivity_results.json
"""

import sys
import os
import json
import pickle
import logging
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

logging.basicConfig(level=logging.WARNING)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "processed", "evaluation")
os.makedirs(OUT_DIR, exist_ok=True)

MACHINE_ID     = "TEAPM_001"
MODEL_DIR      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "data", "processed")
THRESHOLD_FILE = os.path.join(MODEL_DIR, "thresholds.json")
GLITCH_THRESHOLD = 0.2

# Baseline physics confidence parameters
ALPHA_PHYS_FAULT = 0.30
ALPHA_PHYS_WARN  = 0.15
ALPHA_TEMP_BASE  = 0.20
BETA_SPIKE_BASE  = 0.15

# Sensor config inferred from the scaler (mean ± 1σ = normal range)
# Scaler stats: mean, scale
SENSOR_STATS = {
    "ct_01":             {"mean": 55.09,  "scale": 9.15,   "normal_lo": 46.0,  "normal_hi": 64.0},
    "current_sensor_01": {"mean": 324.25, "scale": 90.95,  "normal_lo": 233.0, "normal_hi": 415.0},
    "thermister_01":     {"mean": 47.82,  "scale": 17.66,  "normal_lo": 30.0,  "normal_hi": 65.0},
    "encoder_01":        {"mean": 18509., "scale": 4321.,  "normal_lo": 14188.,"normal_hi": 22830.},
}
SENSOR_COLS = list(SENSOR_STATS.keys())

STATE_TO_LABEL = {"normal": 0, "machine_fault": 1, "sensor_freeze": 1, "sensor_drift": 1, "idle": 1}


# ── SYNTHETIC DATASET ─────────────────────────────────────────────────────────

def generate_synthetic_data(n_normal=14000, n_anomaly=6000, seed=42) -> tuple:
    """
    Generate (X_raw, y_true, states) matching the TEAPM_001 sensor feature space.
    Normal rows: sampled within ±1σ of scaler mean.
    Anomaly rows: sampled at ±(2.5–4)σ — clearly outside normal band.
    Returns raw (unscaled) X, integer labels, and state strings for temporal logic.
    """
    rng = np.random.default_rng(seed)

    # Normal rows
    X_norm_rows = []
    for s, st in SENSOR_STATS.items():
        col = rng.normal(loc=st["mean"], scale=st["scale"] * 0.6, size=n_normal)
        col = np.clip(col, st["normal_lo"] * 0.9, st["normal_hi"] * 1.1)
        X_norm_rows.append(col)
    X_normal = np.column_stack(X_norm_rows)

    # Anomaly rows — split into blocks for realistic temporal structure
    block_types = ["machine_fault", "sensor_drift", "sensor_freeze", "idle"]
    anom_per_type = n_anomaly // len(block_types)
    X_anom_list = []
    states_anom = []
    for btype in block_types:
        rows = []
        for s, st in SENSOR_STATS.items():
            if btype == "idle":
                # Near-zero readings
                col = rng.normal(loc=st["mean"] * 0.1, scale=st["scale"] * 0.1, size=anom_per_type)
            elif btype == "machine_fault":
                # High side deviation
                col = rng.normal(loc=st["mean"] + 3.0 * st["scale"],
                                 scale=st["scale"] * 1.2, size=anom_per_type)
            elif btype == "sensor_drift":
                # Gradual increase
                drift = np.linspace(0, 3.5 * st["scale"], anom_per_type)
                col = st["mean"] + drift + rng.normal(0, st["scale"] * 0.2, anom_per_type)
            else:  # sensor_freeze
                frozen = rng.normal(st["mean"], st["scale"] * 0.05)
                col = np.full(anom_per_type, frozen)
            rows.append(col)
        X_anom_list.append(np.column_stack(rows))
        states_anom.extend([btype] * anom_per_type)

    X_anomaly = np.vstack(X_anom_list)

    # Interleave into blocks (normal/anomaly alternating blocks of 50–150)
    rng2 = np.random.default_rng(seed + 1)
    norm_idx = 0
    anom_idx = 0
    X_rows = []
    state_rows = []
    is_anomaly_block = False

    while norm_idx < n_normal or anom_idx < len(X_anomaly):
        bsize = int(rng2.integers(50, 150))
        if is_anomaly_block and anom_idx < len(X_anomaly):
            end = min(anom_idx + bsize, len(X_anomaly))
            X_rows.append(X_anomaly[anom_idx:end])
            state_rows.extend(states_anom[anom_idx:end])
            anom_idx = end
        elif norm_idx < n_normal:
            end = min(norm_idx + bsize, n_normal)
            X_rows.append(X_normal[norm_idx:end])
            state_rows.extend(["normal"] * (end - norm_idx))
            norm_idx = end
        is_anomaly_block = not is_anomaly_block

    X_raw = np.vstack(X_rows).astype(np.float32)
    states = np.array(state_rows)
    y_true = np.array([STATE_TO_LABEL.get(s, 1) for s in states], dtype=int)
    return X_raw, y_true, states


# ── PHYSICS VIOLATION DETECTION ───────────────────────────────────────────────

def get_physics_violation(raw_row: np.ndarray) -> str:
    """Return 'critical', 'warning', or 'none' for a single raw row."""
    for i, (s, st) in enumerate(SENSOR_STATS.items()):
        val = raw_row[i]
        fault_hi = st["normal_hi"] * 1.20
        fault_lo = st["normal_lo"] * 0.80
        if val > fault_hi or val < fault_lo:
            return "critical"
    for i, (s, st) in enumerate(SENSOR_STATS.items()):
        val = raw_row[i]
        if val > st["normal_hi"] or val < st["normal_lo"]:
            return "warning"
    return "none"


def build_physics_flags(X_raw: np.ndarray) -> np.ndarray:
    flags = np.zeros(len(X_raw), dtype=int)
    for i in range(len(X_raw)):
        v = get_physics_violation(X_raw[i])
        if v == "critical":
            flags[i] = 2
        elif v == "warning":
            flags[i] = 1
    return flags


# ── TEMPORAL PATTERN DETECTION ────────────────────────────────────────────────

def build_temporal_flags(states: np.ndarray, window: int = 5) -> tuple:
    n = len(states)
    labels = np.array([STATE_TO_LABEL.get(s, 1) for s in states])
    is_spike     = np.zeros(n, dtype=bool)
    is_sustained = np.zeros(n, dtype=bool)
    for i in range(n):
        start = max(0, i - window + 1)
        w = labels[start: i + 1]
        cnt = w.sum()
        if labels[i] == 1:
            if cnt == 1:
                is_spike[i] = True
            elif cnt >= 3:
                is_sustained[i] = True
    return is_spike, is_sustained


# ── HYBRID CONFIDENCE ─────────────────────────────────────────────────────────

def compute_hybrid_confidence(
    ml_scores: np.ndarray,
    physics_flags: np.ndarray,
    is_spike: np.ndarray,
    is_sustained: np.ndarray,
    alpha_temp: float,
    beta_spike: float,
) -> np.ndarray:
    c = ml_scores.copy()
    c[physics_flags == 2] += ALPHA_PHYS_FAULT
    c[physics_flags == 1] += ALPHA_PHYS_WARN
    c[is_sustained]        += alpha_temp
    c[is_spike]            -= beta_spike
    return np.clip(c, 0.0, 1.0)


# ── METRICS ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "fpr":       round(fpr, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    import tensorflow as tf
    from models.autoencoder_model import reconstruction_error as dense_error

    print("=" * 65)
    print("  IRAI26 ACTION ITEM #5 — Physics Parameter Sensitivity Analysis")
    print("=" * 65)

    # 1. Load model + scaler + threshold
    model_path = os.path.join(MODEL_DIR, f"autoencoder_{MACHINE_ID}.keras")
    model = tf.keras.models.load_model(model_path)

    scaler_path = os.path.join(MODEL_DIR, f"scaler_{MACHINE_ID}.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(THRESHOLD_FILE) as f:
        thresholds = json.load(f)
    mse_threshold = thresholds[f"{MACHINE_ID}_dense"]
    print(f"\n  Model:     {MACHINE_ID} Dense Autoencoder  (4 sensors)")
    print(f"  MSE threshold: {mse_threshold:.6f}")

    # 2. Generate synthetic dataset matching the model's sensor space
    print("\n  Generating synthetic dataset (20,000 rows, seed=42)…")
    X_raw, y_true, states = generate_synthetic_data(n_normal=14000, n_anomaly=6000, seed=42)
    print(f"  Dataset:   {len(X_raw):,} rows  "
          f"({y_true.sum():,} anomalies, {(y_true==0).sum():,} normal)")

    # 3. Normalise with saved scaler
    import pandas as pd
    X_raw_df = pd.DataFrame(X_raw, columns=SENSOR_COLS)
    X_scaled = scaler.transform(X_raw_df).astype(np.float32)

    # 4. MSE scores (fixed — independent of α/β)
    mse_scores = dense_error(model, X_scaled)

    # 5. Physics flags + temporal flags
    physics_flags = build_physics_flags(X_raw)
    is_spike, is_sustained = build_temporal_flags(states)

    print(f"\n  Physics violations: critical={int((physics_flags==2).sum()):,}  "
          f"warning={int((physics_flags==1).sum()):,}")
    print(f"  Temporal:  spikes={int(is_spike.sum()):,}  "
          f"sustained={int(is_sustained.sum()):,}")

    # 6. Baseline (autoencoder only, no hybrid)
    y_pred_base = (mse_scores > mse_threshold).astype(int)
    baseline = compute_metrics(y_true, y_pred_base)
    print(f"\n  Baseline (autoencoder only, no hybrid filter):")
    print(f"    Precision={baseline['precision']:.4f}  FPR={baseline['fpr']:.4f}  "
          f"F1={baseline['f1']:.4f}")

    # 7. Parameter sweep
    alpha_temp_variants = [
        ("α_temp −50%",   ALPHA_TEMP_BASE * 0.5),
        ("α_temp baseline", ALPHA_TEMP_BASE),
        ("α_temp +50%",   ALPHA_TEMP_BASE * 1.5),
    ]
    beta_spike_variants = [
        ("β_spike −50%",   BETA_SPIKE_BASE * 0.5),
        ("β_spike baseline", BETA_SPIKE_BASE),
        ("β_spike +50%",   BETA_SPIKE_BASE * 1.5),
    ]

    print("\n  Sweep results:")
    print(f"  {'Parameter':<22s} {'Value':>8s} {'Precision':>10s} {'FPR':>8s} "
          f"{'F1':>8s} {'d Prec':>8s} {'d FPR':>8s}")
    print("  " + "-" * 76)

    rows = []

    def run_variant(label, alpha_t, beta_s):
        c_hybrid = compute_hybrid_confidence(
            mse_scores, physics_flags, is_spike, is_sustained, alpha_t, beta_s
        )
        y_pred = y_pred_base.copy()
        glitch_mask = c_hybrid < GLITCH_THRESHOLD
        y_pred[glitch_mask] = 0

        m = compute_metrics(y_true, y_pred)
        d_prec = m["precision"] - baseline["precision"]
        d_fpr  = m["fpr"]       - baseline["fpr"]
        glitch_count = int(glitch_mask.sum())

        tag = " ← baseline" if (alpha_t == ALPHA_TEMP_BASE and beta_s == BETA_SPIKE_BASE) else ""
        val = alpha_t if "temp" in label else beta_s
        print(f"  {label:<22s} {val:>8.3f} "
              f"{m['precision']:>10.4f} {m['fpr']:>8.4f} {m['f1']:>8.4f} "
              f"{d_prec:>+8.4f} {d_fpr:>+8.4f}{tag}")

        return {
            "label": label,
            "alpha_temp": alpha_t,
            "beta_spike": beta_s,
            "glitch_suppressed": glitch_count,
            **m,
            "delta_precision": round(d_prec, 4),
            "delta_fpr":       round(d_fpr, 4),
        }

    for label, alpha_t in alpha_temp_variants:
        rows.append(run_variant(label, alpha_t, BETA_SPIKE_BASE))
    print()
    for label, beta_s in beta_spike_variants:
        rows.append(run_variant(label, ALPHA_TEMP_BASE, beta_s))

    # 8. Summary
    print("\n" + "=" * 65)
    print("  SENSITIVITY SUMMARY")
    print("=" * 65)

    alpha_rows = [r for r in rows if "temp" in r["label"]]
    beta_rows  = [r for r in rows if "spike" in r["label"]]

    def max_delta(row_list, key):
        return max(abs(r[key]) for r in row_list)

    print(f"\n  α_temp sweep (±50%):  max dPrecision={max_delta(alpha_rows,'delta_precision'):+.4f}  "
          f"max dFPR={max_delta(alpha_rows,'delta_fpr'):+.4f}")
    print(f"  β_spike sweep (±50%): max dPrecision={max_delta(beta_rows,'delta_precision'):+.4f}  "
          f"max dFPR={max_delta(beta_rows,'delta_fpr'):+.4f}")
    print("\n  → System is ROBUST if max deltas are < 0.05")

    # 9. Save
    output = {
        "machine_id":  MACHINE_ID,
        "model_type":  "dense",
        "baseline":    baseline,
        "mse_threshold": mse_threshold,
        "sensor_features": SENSOR_COLS,
        "parameters": {
            "alpha_phys_fault": ALPHA_PHYS_FAULT,
            "alpha_phys_warn":  ALPHA_PHYS_WARN,
            "alpha_temp_base":  ALPHA_TEMP_BASE,
            "beta_spike_base":  BETA_SPIKE_BASE,
            "glitch_threshold": GLITCH_THRESHOLD,
        },
        "sweep_results": rows,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = os.path.join(OUT_DIR, "sensitivity_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
