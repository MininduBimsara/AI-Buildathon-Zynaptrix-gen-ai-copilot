import logging
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy import text

# Load environment variables early
load_dotenv()

from unified_rag.api.endpoints import router as rag_router
from unified_rag.db.database import engine, Base, SessionLocal
from unified_rag.db.models import Machine, AnomalyRecord, ChatMessage
from unified_rag.config import settings

from app.websockets import router as websockets_router, broadcast_telemetry
from app.simulator_api import router as simulator_router
from app.machine_api import router as machine_registry_router
from app.assistant_api import router as assistant_router
from app.evaluation_routes import router as evaluation_router
from services.anomaly_tracking import get_tracker

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)

# Database initialization with retries
def init_db():
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
    except Exception as e:
        logger.warning(f"Vector extension issue: {e}")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("✅ Database tables verified successfully.")
            break
        except Exception as e:
            logger.error(f"⚠️ Database connection exception during init (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                logger.critical("❌ Could not connect to the database after retries.")

init_db()

app = FastAPI(title="Generative AI Multi-Agent Industrial Copilot")

# Static assets mounting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
os.makedirs(os.path.join(DATA_DIR, "extracted"), exist_ok=True)
app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")
logger.info(f"✅ Static assets mounted from: {DATA_DIR}")

# CORS Middleware (Strict Security Binding)
allowed_origin = settings.frontend_url.rstrip("/")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[allowed_origin], 
    allow_credentials=True,  # Changed to True to allow secure cookie/socket sharing
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───── API Routes ─────────────────────────────────────────────────────────────

@app.post("/api/telemetry/push")
async def push_telemetry(data: dict):
    """Broadcasts live telemetry to WebSockets and asynchronously runs Anomaly Detection."""
    machine_id = data.get("machine_id", "PUMP-001")
    tracker = get_tracker(machine_id)

    try:
        result = tracker.process(data)
        health_score = result.get("health_score", 100)
    except Exception as e:
        logger.error(f"⚠️ Anomaly analysis failed for {machine_id}: {e}")
        health_score = 100  # Fallback

    broadcast_payload = {
        **data,
        "health_score": health_score
    }
    await broadcast_telemetry(broadcast_payload)
    return {"status": "success", "health": health_score}

# Include routers from sub-modules
app.include_router(rag_router, tags=["Knowledge Base"])
app.include_router(websockets_router, tags=["WebSockets"])
app.include_router(simulator_router)
app.include_router(machine_registry_router)
app.include_router(assistant_router)
app.include_router(evaluation_router, prefix="/api/evaluation", tags=["Model Evaluation"])

# Copilot / Diagnostic specific routes moved to dedicated router
from app.copilot_api import router as copilot_router
app.include_router(copilot_router)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
