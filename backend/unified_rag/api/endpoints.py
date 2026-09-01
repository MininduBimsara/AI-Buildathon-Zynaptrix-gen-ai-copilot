from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel
import os
import tempfile
import time
from datetime import datetime
from typing import Optional
from unified_rag.ingestion.pipeline import process_manual_async
from unified_rag.retrieval.rag import RAGGenerator

router = APIRouter()
rag_gen = RAGGenerator()

class ChatRequest(BaseModel):
    manual_id: str
    query: str

class ChatResponse(BaseModel):
    answer: str
    images: list[str]
    pages: list[int]

@router.post("/ingest-manual")
async def ingest_manual(
    background_tasks: BackgroundTasks,
    manual_id: str = Form(...),
    file: UploadFile = File(...)
):
    print(f"\n🚀 [API] Received ingestion request for Manual ID: {manual_id}")
    print(f"📄 [API] File: {file.filename} (Size roughly: {file.size if hasattr(file, 'size') else 'unknown'} bytes)")

    if not file.filename.endswith(".pdf"):
        print(f"❌ [API] Rejected: {file.filename} is not a PDF")
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()

    # 🌩️ Cloud Sync (Source PDF) — uploaded directly from memory, no local file
    try:
        from services.cloudinary_service import CloudinaryService
        from unified_rag.db.database import SessionLocal
        from unified_rag.db.models import Manual
        from datetime import datetime

        cloud = CloudinaryService()
        if cloud.enabled:
            print(f"☁️ [API] Uploading source PDF for {manual_id} to Cloudinary...")
            url = cloud.upload_file(pdf_bytes, public_id=f"manual_{manual_id}", folder="industrial_copilot/data/manuals", resource_type="raw")

            if url:
                db = SessionLocal()
                try:
                    manual_record = db.query(Manual).filter(Manual.manual_id == manual_id).first()
                    if not manual_record:
                        manual_record = Manual(manual_id=manual_id)
                        db.add(manual_record)
                    manual_record.filename = file.filename
                    manual_record.url = url
                    manual_record.created_at = datetime.now().isoformat()
                    db.commit()
                    print(f"✅ [API] Manual {manual_id} registered in cloud: {url}")
                finally:
                    db.close()
    except Exception as e:
        print(f"⚠️ [API] Source PDF cloud sync failed: {e}")

    # Parsing (PyMuPDF + Camelot) requires a real filesystem path — Camelot in
    # particular has no in-memory API. Use a momentary OS-tempdir file (not
    # backend/data/), always deleted once the pipeline finishes.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    # Run the pipeline AFTER responding. A large manual makes one vision call per
    # figure — a 144-page document ran for many minutes, long enough for the
    # browser to abandon the request and show "Failed to fetch" even though the
    # server was still working. Poll /ingest-manual/status/{manual_id} instead.
    _set_job(manual_id, status="processing", filename=file.filename)
    background_tasks.add_task(_run_ingestion, tmp_path, manual_id)
    print(f"✅ [API] File received. Ingestion for {manual_id} queued in background.")

    return {
        "message": "Ingestion started",
        "manual_id": manual_id,
        "status": "processing",
        "poll": f"/ingest-manual/status/{manual_id}",
    }


# ── Background ingestion job tracking ────────────────────────────────────────
# In-process registry; the API runs as a single uvicorn process and this state is
# advisory progress reporting, not a source of truth (the chunks themselves land
# in Postgres).
_ingestion_jobs: dict[str, dict] = {}


def _set_job(manual_id: str, **fields) -> None:
    job = _ingestion_jobs.setdefault(manual_id, {"manual_id": manual_id})
    job.update(fields)
    job["updated_at"] = datetime.now().isoformat()


async def _run_ingestion(tmp_path: str, manual_id: str) -> None:
    """Execute the ingestion pipeline outside the request/response cycle."""
    try:
        result = await process_manual_async(tmp_path, manual_id)
        _set_job(manual_id, status="success", chunks=result.get("chunks"), error=None)
        print(f"🏁 [API] Ingestion successful for {manual_id}!")
    except Exception as e:
        print(f"🔥 [API] CRITICAL ERROR during ingestion: {str(e)}")
        import traceback
        traceback.print_exc()
        _set_job(manual_id, status="failed", error=str(e))
    finally:
        # Best-effort only. On Windows the PDF backends (PyMuPDF/Camelot/playa) can
        # still hold the handle when we get here, and os.remove then raises
        # PermissionError [WinError 32]. Cleanup must never mask the result.
        _cleanup_temp_pdf(tmp_path)


@router.get("/ingest-manual/status/{manual_id}")
async def ingestion_status(manual_id: str):
    """Progress for a queued ingestion. 'unknown' once the server has restarted."""
    return _ingestion_jobs.get(
        manual_id, {"manual_id": manual_id, "status": "unknown"}
    )


def _cleanup_temp_pdf(tmp_path: Optional[str]) -> None:
    """Delete the scratch PDF, retrying briefly while a parser releases its handle."""
    if not tmp_path or not os.path.exists(tmp_path):
        return
    for attempt in range(5):
        try:
            os.remove(tmp_path)
            return
        except PermissionError:
            time.sleep(0.2 * (attempt + 1))
        except OSError as e:
            print(f"⚠️ [API] Could not remove temp file {tmp_path}: {e}")
            return
    print(f"⚠️ [API] Temp file still locked, leaving for OS cleanup: {tmp_path}")

from unified_rag.db.database import SessionLocal
from unified_rag.db.models import Machine
from sqlalchemy.orm import Session
from fastapi import Depends

# ... (existing imports)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class MachineCreate(BaseModel):
    machine_id: str
    name: str
    location: str
    manual_id: str

class MachineResponse(BaseModel):
    machine_id: str
    name: str
    location: str
    manual_id: str
    class Config:
        from_attributes = True

@router.post("/machines", response_model=MachineResponse)
async def create_machine(machine: MachineCreate, db: Session = Depends(get_db)):
    # Check if machine already exists to avoid 500/IntegrityError
    existing = db.query(Machine).filter(Machine.machine_id == machine.machine_id).first()
    if existing:
        # Update existing record if needed
        for key, value in machine.model_dump().items():
            setattr(existing, key, value)
        db.commit()
        db.refresh(existing)
        return existing
        
    db_machine = Machine(**machine.model_dump())
    db.add(db_machine)
    db.commit()
    db.refresh(db_machine)
    return db_machine

@router.get("/machines", response_model=list[MachineResponse])
async def list_machines(db: Session = Depends(get_db)):
    return db.query(Machine).all()

@router.post("/machines/delete/{machine_id}")
async def delete_machine(machine_id: str, db: Session = Depends(get_db)):
    print(f"🗑️ [API] Deletion request (POST) for Machine ID: {machine_id}")
    machine = db.query(Machine).filter(Machine.machine_id == machine_id).first()
    if not machine:
        print(f"⚠️ [API] Machine {machine_id} not found for deletion")
        raise HTTPException(status_code=404, detail="Machine not found")
    
    db.delete(machine)
    db.commit()
    print(f"✅ [API] Machine {machine_id} successfully decommissioned")
    return {"message": f"Machine {machine_id} deleted successfully"}

@router.get("/machines/{machine_id}", response_model=MachineResponse)
async def get_machine(machine_id: str, db: Session = Depends(get_db)):
    machine = db.query(Machine).filter(Machine.machine_id == machine_id).first()
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    return machine

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response_data = rag_gen.generate_response(request.query, request.manual_id)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
