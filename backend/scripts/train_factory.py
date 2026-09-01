"""
train_factory.py — Orchestrates the full Generate -> Preprocess -> Train pipeline
for the entire 3-machine registry in one command.

Ensures that every machine in the factory has a high-fidelity anomaly detector.
"""

import subprocess
import sys
import os
import time

# Machine Registry
MACHINES = ["PUMP-001", "LATHE-002", "TURBINE-003"]

def run_step(command: list, description: str):
    print(f"\n🚀 {description} ...")
    start = time.time()
    result = subprocess.run([sys.executable] + command, capture_output=False)
    if result.returncode != 0:
        print(f"❌ Step failed: {description}")
        sys.exit(1)
    duration = time.time() - start
    print(f"✅ Completed in {duration:.1f}s")

def train_factory():
    print("═" * 60)
    print("  INDUSTRIAL COPILOT: FACTORY-WIDE ANOMALY TRAINING")
    print("═" * 60)

    for machine_id in MACHINES:
        print(f"\n🤖 Processing {machine_id} ...")
        
        # 1. Generate Raw Data (20,000 samples per machine)
        run_step(
            ["generate_dataset.py", "--machine_id", machine_id, "--rows", "20000"],
            f"Generating synthetic training data for {machine_id}"
        )

        # 2. Normalize (Fit per-machine scaler)
        run_step(
            ["preprocessing/normalization.py", "--machine_id", machine_id],
            f"Fitting per-machine scaler for {machine_id}"
        )

        # 3. Train Model (Autoencoder)
        run_step(
            ["models/train_model.py", "--model", "dense", "--machine_id", machine_id],
            f"Training Dense Autoencoder for {machine_id}"
        )

    print("\n" + "═" * 60)
    print("🎉 FACTORY-WIDE TRAINING COMPLETE!")
    print(f"   Assets processed: {len(MACHINES)}")
    print("   Models, scalers, thresholds: Cloudinary + Neon (MachineAsset / AnomalyThreshold)")
    print("═" * 60)

if __name__ == "__main__":
    train_factory()
