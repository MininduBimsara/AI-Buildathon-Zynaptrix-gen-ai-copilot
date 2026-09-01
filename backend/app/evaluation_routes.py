"""
evaluation_routes.py — FastAPI endpoints for model evaluation metrics and plots.

Backed entirely by Neon (MachineEvaluation table) + Cloudinary plot URLs —
no local files are read.

Endpoints:
    GET  /api/evaluation/{machine_id}                    → Metrics JSON
    GET  /api/evaluation/{machine_id}/plots/{plot_name}  → Redirect to plot's Cloudinary URL
    POST /api/evaluation/{machine_id}/run                → Trigger fresh evaluation
    GET  /api/evaluation/summary                         → All machines summary
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse

from unified_rag.db.database import SessionLocal
from unified_rag.db.models import MachineEvaluation

log = logging.getLogger(__name__)
router = APIRouter()


def _load_evaluation(machine_id: str, model_type: str = "dense") -> dict:
    """Load the stored evaluation for a machine/model from Neon."""
    db = SessionLocal()
    try:
        record = db.query(MachineEvaluation).filter(
            MachineEvaluation.machine_id == machine_id,
            MachineEvaluation.model_type == model_type
        ).first()
        if not record:
            raise FileNotFoundError(f"No evaluation found for {machine_id}/{model_type}")

        return {
            "machine_id": record.machine_id,
            "model_type": record.model_type,
            "metrics": json.loads(record.metrics_json),
            "plot_urls": json.loads(record.plot_urls_json),
            "evaluated_at": record.evaluated_at,
        }
    finally:
        db.close()


# ── GET /api/evaluation/summary ──────────────────────────────────────────────

@router.get("/summary")
async def get_evaluation_summary():
    """Returns evaluation summary for all machines, queried directly from Neon."""
    db = SessionLocal()
    try:
        records = db.query(MachineEvaluation).all()
    finally:
        db.close()

    summary_rows = []
    for record in records:
        try:
            m = json.loads(record.metrics_json)
            summary_rows.append({
                "machine_id":       record.machine_id,
                "model_type":       record.model_type,
                "accuracy":         m.get("accuracy"),
                "precision":        m.get("precision"),
                "recall":           m.get("recall"),
                "f1_score":         m.get("f1_score"),
                "auc_roc":          m.get("auc_roc"),
                "fpr":              m.get("false_positive_rate"),
                "fnr":              m.get("false_negative_rate"),
                "threshold":        m.get("threshold"),
                "separation_ratio": m.get("separation_ratio"),
                "evaluated_at":     record.evaluated_at,
            })
        except Exception as e:
            log.warning(f"Error parsing evaluation for {record.machine_id}/{record.model_type}: {e}")

    return {"status": "success", "data": {"summary": summary_rows}}


# ── GET /api/evaluation/{machine_id} ─────────────────────────────────────────

@router.get("/{machine_id}")
async def get_evaluation(
    machine_id: str,
    model_type: str = Query("dense", description="Model type: dense or lstm")
):
    """Returns evaluation metrics + plot URLs for a specific machine and model type."""
    try:
        data = _load_evaluation(machine_id, model_type)
        return {"status": "success", "data": data}

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No evaluation found for {machine_id}/{model_type}. "
                   f"Run POST /api/evaluation/{machine_id}/run first."
        )
    except Exception as e:
        log.error(f"Error loading evaluation for {machine_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /api/evaluation/{machine_id}/plots/{plot_name} ────────────────────────

@router.get("/{machine_id}/plots/{plot_name}")
async def get_evaluation_plot(machine_id: str, plot_name: str):
    """
    Redirects to the plot's Cloudinary URL. Kept for backward-compatibility with
    the old local-file-serving URL pattern; new code should just use `plot_urls`
    from GET /api/evaluation/{machine_id} directly.

    plot_name examples (legacy filename shape):
        roc_curve_dense.png
        mse_distribution_lstm.png
    """
    stem = plot_name.removesuffix(".png")
    model_type = "lstm" if stem.endswith("_lstm") else "dense"
    plot_type = stem.removesuffix(f"_{model_type}")

    try:
        data = _load_evaluation(machine_id, model_type)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No evaluation found for {machine_id}/{model_type}")

    url = data["plot_urls"].get(plot_type)
    if not url:
        raise HTTPException(status_code=404, detail=f"Plot not found: {plot_name} for {machine_id}")

    return RedirectResponse(url)


# ── POST /api/evaluation/{machine_id}/run ─────────────────────────────────────

@router.post("/{machine_id}/run")
async def run_evaluation(
    machine_id: str,
    model_type: str = Query("dense", description="Model type: dense, lstm, or both")
):
    """
    Triggers a fresh evaluation for the specified machine.
    This may take 10-30 seconds depending on dataset size.
    """
    try:
        from models.evaluate_model import evaluate_autoencoder

        results = {}
        model_types = ["dense", "lstm"] if model_type == "both" else [model_type]

        for mt in model_types:
            try:
                result = evaluate_autoencoder(machine_id, mt)
                results[mt] = result
            except FileNotFoundError as e:
                results[mt] = {"error": str(e), "status": "skipped"}
            except Exception as e:
                log.error(f"Evaluation failed for {machine_id}/{mt}: {e}")
                results[mt] = {"error": str(e), "status": "failed"}

        return {
            "status": "success",
            "message": f"Evaluation complete for {machine_id}",
            "results": results
        }

    except Exception as e:
        log.error(f"Evaluation error for {machine_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
