import sys
import logging
import subprocess
from typing import Dict, List
from fastapi import APIRouter, HTTPException

from simulator.anomaly_injector import get_machine_config as get_default

router = APIRouter(prefix="/api/simulator", tags=["Simulator"])
logger = logging.getLogger(__name__)

simulator_processes: Dict[str, subprocess.Popen] = {}


# The frontend interpolates the selected machine straight into the query string,
# so an unset selection arrives as the literal text "null"/"undefined". That used
# to spawn a real simulator process for a machine that has no model, which then
# logged "No cloud-registered model/scaler found for null" on every reading.
_PLACEHOLDER_IDS = {"", "null", "undefined", "none", "nan"}


def _require_machine_id(machine_id: str) -> str:
    cleaned = (machine_id or "").strip()
    if cleaned.lower() in _PLACEHOLDER_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid machine_id {machine_id!r}. Select a machine before starting the simulator.",
        )
    return cleaned


@router.post("/start")
async def start_simulator(machine_id: str = "PUMP-001"):
    global simulator_processes
    machine_id = _require_machine_id(machine_id)
    if machine_id not in simulator_processes or simulator_processes[machine_id].poll() is not None:
        proc = subprocess.Popen([sys.executable, "-m", "simulator.sensor_simulator", "--machine_id", machine_id])
        simulator_processes[machine_id] = proc
        logger.info(f"🚀 Started simulator for {machine_id} (PID: {proc.pid})")
        return {"status": "started", "machine_id": machine_id, "pid": proc.pid}
    return {"status": "already_running", "machine_id": machine_id}

@router.post("/stop")
async def stop_simulator(machine_id: str = "PUMP-001"):
    global simulator_processes
    if machine_id in simulator_processes and simulator_processes[machine_id].poll() is None:
        proc = simulator_processes[machine_id]
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        del simulator_processes[machine_id]
        logger.info(f"🛑 Stopped simulator for {machine_id}")
        return {"status": "stopped", "machine_id": machine_id}
    return {"status": "not_running", "machine_id": machine_id}

@router.post("/inject")
async def inject_simulator(machine_id: str = "PUMP-001", anomaly_type: str = "machine_fault"):
    machine_id = _require_machine_id(machine_id)
    state_file = f"simulator_{machine_id}_override.state"
    with open(state_file, "w") as f:
         f.write(anomaly_type)
    logger.info(f"💉 Injecting forced {anomaly_type} for {machine_id}")
    return {"status": "injected", "machine_id": machine_id, "anomaly_type": anomaly_type}

@router.get("/status")
async def get_simulator_status():
    """Returns a list of machine IDs that are currently being simulated."""
    active = [mid for mid, proc in simulator_processes.items() if proc.poll() is None]
    return {"active_simulators": active}

@router.get("/machine/{machine_id}/config")
async def get_machine_config(machine_id: str):
    """Return the registered sensor IDs + icon_type for this machine, or defaults."""
    from services.sensor_config_loader import sensor_config_loader

    machine_cfg = sensor_config_loader.get_machine_config(machine_id)
    if machine_cfg:
        sensors_meta = [
            {
                "sensor_id":   sid,
                "sensor_name": sdata.get("sensor_name", sid),
                "icon_type":   sdata.get("icon_type", "generic"),
                "unit":        sdata.get("unit", "units"),
            }
            for sid, sdata in machine_cfg.items()
        ]
        return {"sensors": [s["sensor_id"] for s in sensors_meta], "sensors_meta": sensors_meta}

    # Fallback to defaults
    default_cfg = get_default(machine_id)
    sensors_meta = [
        {"sensor_id": sid, "sensor_name": sid, "icon_type": "generic", "unit": "units"}
        for sid in default_cfg.keys()
    ]
    return {"sensors": list(default_cfg.keys()), "sensors_meta": sensors_meta}
