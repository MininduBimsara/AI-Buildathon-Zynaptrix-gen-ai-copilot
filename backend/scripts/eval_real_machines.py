"""
eval_real_machines.py — Run full evaluation on real PUMP-001, LATHE-002, TURBINE-003 models.

Generates synthetic data matching each machine's sensor space, runs the real Cloudinary
autoencoder, computes metrics, and regenerates all plots including Figure 6 (log scale).

Usage (from backend/):
    & "C:\\ProgramData\\miniconda3\\python.exe" scripts/eval_real_machines.py
"""

import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

# Suppress TF noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "processed")
EVAL_DIR  = os.path.join(MODEL_DIR, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

# Real sensor specs per machine (from simulator configs / scaler means)
MACHINE_CONFIGS = {
    "PUMP-001": {
        "threshold": 0.6994112730026245,
        "sensors": {
            "temperature":   {"mean": 180.0, "std": 2.0,   "normal_lo": 175.0, "normal_hi": 185.0},
            "motor_current": {"mean": 4.5,   "std": 0.5,   "normal_lo": 3.0,   "normal_hi": 6.0},
            "vibration":     {"mean": 0.80,  "std": 0.10,  "normal_lo": 0.5,   "normal_hi": 1.2},
            "speed":         {"mean": 160.0, "std": 5.0,   "normal_lo": 150.0, "normal_hi": 170.0},
            "pressure":      {"mean": 4.5,   "std": 0.2,   "normal_lo": 4.0,   "normal_hi": 5.0},
        },
    },
    "LATHE-002": {
        "threshold": 0.7203473448753357,
        "sensors": {
            "temperature":   {"mean": 45.0,    "std": 3.0,   "normal_lo": 35.0,   "normal_hi": 55.0},
            "motor_current": {"mean": 12.5,    "std": 1.5,   "normal_lo": 8.0,    "normal_hi": 18.0},
            "vibration":     {"mean": 0.15,    "std": 0.03,  "normal_lo": 0.05,   "normal_hi": 0.30},
            "speed":         {"mean": 3200.0,  "std": 200.0, "normal_lo": 2500.0, "normal_hi": 4000.0},
            "pressure":      {"mean": 8.5,     "std": 0.5,   "normal_lo": 6.0,    "normal_hi": 11.0},
        },
    },
    "TURBINE-003": {
        "threshold": 0.7300003170967102,
        "sensors": {
            "temperature":   {"mean": 850.0,   "std": 20.0,  "normal_lo": 800.0,  "normal_hi": 920.0},
            "motor_current": {"mean": 450.0,   "std": 30.0,  "normal_lo": 380.0,  "normal_hi": 530.0},
            "vibration":     {"mean": 1.20,    "std": 0.15,  "normal_lo": 0.8,    "normal_hi": 1.8},
            "speed":         {"mean": 15000.0, "std": 500.0, "normal_lo": 13000.0,"normal_hi": 17000.0},
            "pressure":      {"mean": 32.0,    "std": 2.0,   "normal_lo": 26.0,   "normal_hi": 38.0},
        },
    },
}

STATE_TO_LABEL = {"normal": 0, "machine_fault": 1, "sensor_drift": 1, "sensor_freeze": 1, "idle": 1}
COLORS = {"success": "#2ecc71", "danger": "#e74c3c", "dark": "#2c3e50"}


def generate_data(cfg, n_normal=14000, n_anomaly=6000, seed=42):
    sensors = cfg["sensors"]
    cols = list(sensors.keys())
    rng = np.random.default_rng(seed)

    # Normal rows — within ±1σ of mean
    Xn = np.column_stack([
        np.clip(rng.normal(s["mean"], s["std"] * 0.6, n_normal),
                s["normal_lo"] * 0.9, s["normal_hi"] * 1.1)
        for s in sensors.values()
    ])

    # Anomaly rows — 4 types
    anom_per = n_anomaly // 4
    blocks, states = [], []
    for btype in ["machine_fault", "sensor_drift", "sensor_freeze", "idle"]:
        cols_data = []
        for s in sensors.values():
            if btype == "idle":
                col = rng.normal(s["mean"] * 0.1, s["std"] * 0.1, anom_per)
            elif btype == "machine_fault":
                col = rng.normal(s["mean"] + 3.2 * s["std"], s["std"] * 1.4, anom_per)
            elif btype == "sensor_drift":
                drift = np.linspace(0, 3.5 * s["std"], anom_per)
                col = s["mean"] + drift + rng.normal(0, s["std"] * 0.2, anom_per)
            else:  # freeze
                col = np.full(anom_per, rng.normal(s["mean"], s["std"] * 0.02))
            cols_data.append(col)
        blocks.append(np.column_stack(cols_data))
        states.extend([btype] * anom_per)
    Xa = np.vstack(blocks)

    # Interleave into temporal blocks
    rng2 = np.random.default_rng(seed + 1)
    ni, ai, X_rows, st_rows, is_anom = 0, 0, [], [], False
    while ni < n_normal or ai < len(Xa):
        bsize = int(rng2.integers(50, 150))
        if is_anom and ai < len(Xa):
            end = min(ai + bsize, len(Xa))
            X_rows.append(Xa[ai:end]); st_rows.extend(states[ai:end]); ai = end
        elif ni < n_normal:
            end = min(ni + bsize, n_normal)
            X_rows.append(Xn[ni:end]); st_rows.extend(["normal"] * (end - ni)); ni = end
        is_anom = not is_anom

    X = np.vstack(X_rows).astype(np.float32)
    y = np.array([STATE_TO_LABEL.get(s, 1) for s in st_rows], dtype=int)
    return X, y, np.array(st_rows), cols


def run_evaluation(machine_id):
    cfg = MACHINE_CONFIGS[machine_id]
    threshold = cfg["threshold"]

    print(f"\n{'='*60}")
    print(f"  {machine_id}")
    print(f"{'='*60}")

    # Load model + scaler
    model_path  = os.path.join(MODEL_DIR, f"autoencoder_{machine_id}.keras")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{machine_id}.pkl")
    model = tf.keras.models.load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Generate data
    X_raw, y_true, states, cols = generate_data(cfg)
    X_scaled = scaler.transform(pd.DataFrame(X_raw, columns=cols)).astype(np.float32)

    # MSE scores
    X_pred = model.predict(X_scaled, verbose=0)
    mse = np.mean(np.square(X_scaled - X_pred), axis=1)

    # Classification
    y_pred = (mse > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    auc       = roc_auc_score(y_true, mse)
    fpr_val   = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    normal_mse = mse[y_true == 0]
    fault_mse  = mse[y_true == 1]
    sep_ratio  = np.mean(fault_mse) / max(np.mean(normal_mse), 1e-9)

    print(f"  Threshold:  {threshold:.4f}")
    print(f"  µ_normal:   {np.mean(normal_mse):.4f}  (σ={np.std(normal_mse):.4f})")
    print(f"  µ_fault:    {np.mean(fault_mse):.4f}  (σ={np.std(fault_mse):.4f})")
    print(f"  Separation: {sep_ratio:.1f}×")
    print(f"  Precision:  {precision*100:.2f}%  Recall: {recall*100:.2f}%  F1: {f1*100:.2f}%")
    print(f"  AUC-ROC:    {auc:.4f}   FPR: {fpr_val*100:.2f}%")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    # Save Figure 6 (MSE distribution, log scale)
    out_dir = os.path.join(EVAL_DIR, machine_id)
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    all_mse = np.concatenate([normal_mse, fault_mse])
    pos_mse = all_mse[all_mse > 0]
    min_val = max(pos_mse.min() * 0.5, 1e-6)
    max_val = np.percentile(pos_mse, 99.5)
    bins = np.logspace(np.log10(min_val), np.log10(max_val), 80)

    ax.hist(normal_mse[normal_mse > 0], bins=bins,
            color=COLORS["success"], alpha=0.65, edgecolor="white", density=True,
            label=f"Normal (n={len(normal_mse):,}, µ={np.mean(normal_mse):.4f})")
    ax.hist(fault_mse[fault_mse > 0], bins=bins,
            color=COLORS["danger"], alpha=0.55, edgecolor="white", density=True,
            label=f"Fault/Anomaly (n={len(fault_mse):,}, µ={np.mean(fault_mse):.4f})")
    ax.set_xscale("log")
    ax.axvline(threshold, color=COLORS["dark"], linewidth=2.5, linestyle="--",
               label=f"Threshold = {threshold:.4f}")
    ax.axvline(np.mean(normal_mse), color=COLORS["success"], linewidth=1.5, linestyle=":", alpha=0.8)
    ax.axvline(np.mean(fault_mse),  color=COLORS["danger"],  linewidth=1.5, linestyle=":", alpha=0.8)
    ax.set_xlabel("Reconstruction Error (MSE) — log scale", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"MSE Distribution (Log Scale) — {machine_id} (Dense Autoencoder)\n"
        f"Normal µ={np.mean(normal_mse):.4f} vs Fault µ={np.mean(fault_mse):.4f} "
        f"— {sep_ratio:.0f}× Separation",
        fontsize=13)
    ax.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()
    fig6_path = os.path.join(out_dir, "mse_distribution_dense.png")
    plt.savefig(fig6_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 6 saved → {fig6_path}")

    return {
        "machine_id": machine_id,
        "threshold": threshold,
        "mu_normal": round(float(np.mean(normal_mse)), 6),
        "mu_fault":  round(float(np.mean(fault_mse)), 6),
        "separation_ratio": round(sep_ratio, 1),
        "precision":  round(precision, 4),
        "recall":     round(recall, 4),
        "f1":         round(f1, 4),
        "auc_roc":    round(auc, 4),
        "fpr":        round(fpr_val, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


if __name__ == "__main__":
    all_results = {}
    for mid in ["PUMP-001", "LATHE-002", "TURBINE-003"]:
        all_results[mid] = run_evaluation(mid)

    out_path = os.path.join(EVAL_DIR, "real_machine_evaluation.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nAll results saved → {out_path}")

    print("\n\n=== SUMMARY FOR PAPER ===")
    print(f"{'Machine':<15} {'µ_normal':>10} {'µ_fault':>10} {'Sep':>7} {'Prec':>8} {'F1':>8} {'AUC':>8} {'FPR':>8}")
    print("-" * 75)
    for mid, r in all_results.items():
        print(f"{mid:<15} {r['mu_normal']:>10.4f} {r['mu_fault']:>10.4f} "
              f"{r['separation_ratio']:>6.0f}x {r['precision']:>8.4f} "
              f"{r['f1']:>8.4f} {r['auc_roc']:>8.4f} {r['fpr']:>8.4f}")
