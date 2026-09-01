import json
import logging
from typing import List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
logger = logging.getLogger(__name__)

class TelemetryClientManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.debug(f"Failed to broadcast to connection: {e}")
                pass

telemetry_manager = TelemetryClientManager()

@router.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await telemetry_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        telemetry_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        telemetry_manager.disconnect(websocket)

async def broadcast_anomaly(data: dict):
    await telemetry_manager.broadcast(json.dumps({
        "type": "anomaly_alert",
        "data": data
    }))

async def broadcast_telemetry(data: dict):
    await telemetry_manager.broadcast(json.dumps({
        "type": "telemetry",
        "data": data
    }))
