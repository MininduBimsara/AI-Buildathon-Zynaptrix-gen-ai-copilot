from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from unified_rag.db.database import SessionLocal
from unified_rag.db.models import Machine, AnomalyRecord, ChatMessage, SensorConfiguration
from unified_rag.ai_client import get_client, MODEL_CHAT
from pydantic import BaseModel
import json
import os
import sys
from datetime import datetime
from fastapi import Request, UploadFile, BackgroundTasks
import subprocess
import asyncio
from services.datasheet_parser import DatasheetParser
from services.incident_summarizer import summarize_and_archive

router = APIRouter(tags=["Machine Registry"])

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic Schemas
class MachineBase(BaseModel):
    machine_id: str
    name: str
    location: str
    manual_id: str

class MachineResponse(MachineBase):
    class Config:
        from_attributes = True

class AnomalyResponse(BaseModel):
    id: int
    machine_id: str
    timestamp: str
    type: str
    score: float
    sensor_data: str # JSON backend
    resolved: bool

    class Config:
        from_attributes = True

class ResolveRequest(BaseModel):
    operator_fix: str

@router.get("/api/chat-history/{anomaly_id}")
async def get_chat_history(anomaly_id: int):
    db = SessionLocal()
    try:
        messages = db.query(ChatMessage).filter(ChatMessage.anomaly_id == anomaly_id).order_by(ChatMessage.id).all()
        return [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
                "images": json.loads(m.images) if m.images else [],
                "metadata": json.loads(m.message_metadata) if m.message_metadata else None,
                "db_id": m.id
            } 
            for m in messages
        ]
    finally:
        db.close()

class TaskUpdateRequest(BaseModel):
    task_id: str
    completed: bool

@router.patch("/api/chat-message/{message_id}/task")
async def update_task_status(message_id: int, req: TaskUpdateRequest):
    """Update a single task completion status within a procedure stored in a chat message."""
    db = SessionLocal()
    try:
        msg = db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
        if not msg:
            raise HTTPException(status_code=404, detail="Message not found")
        
        meta = json.loads(msg.message_metadata) if msg.message_metadata else {}
        if "completed_tasks" not in meta:
            meta["completed_tasks"] = {}
        meta["completed_tasks"][req.task_id] = req.completed
        msg.message_metadata = json.dumps(meta)
        db.commit()
        return {"status": "updated", "completed_tasks": meta["completed_tasks"]}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@router.post("/api/chat-history/{anomaly_id}/resolve")
async def resolve_incident(anomaly_id: int, req: ResolveRequest):
    """
    Resolve an incident with AI-validated operator feedback.
    
    Workflow:
    1. AI Automation Engineer validates the feedback quality
    2. If valid, summarize and vectorize for RAG
    3. Archive with thank you message and assistant link
    """
    db = SessionLocal()
    try:
        # 1. Find and verify the anomaly record
        record = db.query(AnomalyRecord).filter(AnomalyRecord.id == anomaly_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Incident not found")
        
        # 2. AI Validation of feedback using AI Automation Engineer
        validation_result = await validate_operator_feedback(
            operator_fix=req.operator_fix,
            machine_id=record.machine_id,
            anomaly_type=record.type,
            ai_validation_status=getattr(record, 'ai_validation_status', None)
        )
        
        if not validation_result.get("is_valid", True):
            # Feedback doesn't meet quality standards - return suggestion
            return {
                "status": "validation_failed",
                "message": validation_result.get("feedback_message", "Please provide more detailed feedback about the actions taken."),
                "suggestions": validation_result.get("suggestions", [])
            }
        
        # 3. Mark Anomaly as Resolved
        record.resolved = True
        
        # 4. Extract Chat History
        messages = db.query(ChatMessage).filter(ChatMessage.anomaly_id == anomaly_id).all()
        history_text = "\n".join([f"{m.role}: {m.content}" for m in messages])
        
        # 5-7. Summarize, embed, and archive into InteractionMemory (for RAG)
        summary = summarize_and_archive(history_text, req.operator_fix, record.machine_id, db)

        # 8. Generate thank you message with AI enhancement
        thank_you_message = generate_thank_you_message(
            operator_fix=req.operator_fix,
            machine_id=record.machine_id,
            validation_result=validation_result
        )

        # 9. Store as a ChatMessage (for word-for-word history & archival view)
        summary_msg = ChatMessage(
            anomaly_id=anomaly_id,
            role='agent',
            content=thank_you_message,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            message_metadata=json.dumps({"type": "final_summary", "ai_validated": True})
        )
        db.add(summary_msg)
        
        db.commit()
        return {
            "status": "resolved_and_archived",
            "summary": summary,
            "thank_you_message": thank_you_message,
            "ai_validation": validation_result
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


async def validate_operator_feedback(
    operator_fix: str,
    machine_id: str,
    anomaly_type: str,
    ai_validation_status: str = None
) -> dict:
    """
    Validate operator feedback using AI Automation Engineer.
    
    Checks for:
    - Minimum detail/quality of feedback
    - Technical relevance to the anomaly type
    - Actionable information for future reference
    """
    try:
        from agents.ai_automation_engineer import AIAutomationEngineerAgent

        ai_engineer = AIAutomationEngineerAgent()
        
        # Build validation prompt
        validation_prompt = f"""
        Validate this operator feedback for archival quality.
        
        Machine: {machine_id}
        Anomaly Type: {anomaly_type}
        AI Classification: {ai_validation_status or 'Not classified'}
        
        Operator Feedback:
        "{operator_fix}"
        
        Evaluate:
        1. Is this feedback detailed enough to be useful for future troubleshooting?
        2. Does it describe actual actions taken (not just observations)?
        3. Is it technically relevant to the anomaly type?
        
        Return JSON:
        {{
            "is_valid": true/false,
            "quality_score": 0.0-1.0,
            "feedback_message": "Message to show operator if invalid",
            "suggestions": ["List of suggestions to improve feedback"],
            "extracted_actions": ["List of actions identified"],
            "improvement_areas": "Brief note on what could be added"
        }}
        """
        
        response = ai_engineer.client.chat.completions.create(
            model=MODEL_CHAT,
            messages=[
                {"role": "system", "content": "You are an AI Automation Engineer validating maintenance feedback quality."},
                {"role": "user", "content": validation_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Apply minimum quality threshold (0.3 = very permissive)
        min_quality = 0.3
        if result.get("quality_score", 0) < min_quality and len(operator_fix.strip()) < 20:
            result["is_valid"] = False
            result["feedback_message"] = "Please provide a bit more detail about the actions you took. Even a short description helps future troubleshooting."
        else:
            result["is_valid"] = True
            
        return result
        
    except Exception as e:
        # On error, allow the feedback through but note the validation failure
        return {
            "is_valid": True,
            "quality_score": 0.5,
            "validation_error": str(e),
            "feedback_message": None,
            "suggestions": []
        }


def generate_thank_you_message(
    operator_fix: str,
    machine_id: str,
    validation_result: dict
) -> str:
    """
    Generate a thank you message with link to central assistant.
    """
    quality_score = validation_result.get("quality_score", 0.7)
    extracted_actions = validation_result.get("extracted_actions", [])
    
    # Build action summary if available
    action_summary = ""
    if extracted_actions:
        action_summary = f"\n\n**Actions Recorded:**\n" + "\n".join([f"• {action}" for action in extracted_actions[:3]])
    
    # Quality-based appreciation
    if quality_score >= 0.8:
        appreciation = "🌟 **Excellent documentation!** Your detailed feedback will significantly help future troubleshooting."
    elif quality_score >= 0.6:
        appreciation = "✅ **Great feedback!** This information has been added to our knowledge base."
    else:
        appreciation = "👍 **Thank you!** Your feedback has been recorded."
    
    message = f"""### 🏁 Incident Archived Successfully

{appreciation}
{action_summary}

**Machine:** {machine_id}
**Operator Notes:** {operator_fix}

---

📋 **Your feedback has been:**
• ✓ AI-validated for quality
• ✓ Vectorized into the knowledge base
• ✓ Linked to this incident for future reference

---

💬 **Need more help?**
If you encounter any issues or have questions, chat with the **Central Assistant** anytime:

🔗 [Open Central Assistant](/assistant) | Type your question in the main chat

_Thank you for keeping our systems running smoothly!_ 🔧"""

    return message

@router.get("/api/machines", response_model=List[MachineResponse])
async def list_machines(db: Session = Depends(get_db)):
    return db.query(Machine).all()

@router.get("/api/machines/{machine_id}/anomalies", response_model=List[AnomalyResponse])
async def get_machine_anomalies(machine_id: str, db: Session = Depends(get_db)):
    """Fetch history of anomalies for a specific machine."""
    return db.query(AnomalyRecord).filter(AnomalyRecord.machine_id == machine_id).order_by(AnomalyRecord.id.desc()).all()

@router.post("/api/machines", response_model=MachineResponse)
async def register_machine(request: Request, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Add or update a machine in the registry via FormData (supports PDFs).
    
    Enhanced workflow with AI Automation Engineer:
    1. Parse sensor datasheets with AI extraction
    2. Validate configs with AI Automation Engineer
    3. Cross-validate sensor relationships
    4. Generate AI-powered anomaly patterns for training data
    5. Train ML model with realistic patterns
    """
    import logging
    logger = logging.getLogger(__name__)
    
    form_data = await request.form()
    
    machine_id = str(form_data.get('machine_id'))
    name = str(form_data.get('name') or "")
    location = str(form_data.get('location') or "")
    manual_id = str(form_data.get('manual_id') or "")

    if not machine_id or machine_id == "None":
        raise HTTPException(status_code=400, detail="machine_id is required")

    machine_dict = {
        "machine_id": machine_id,
        "name": name,
        "location": location,
        "manual_id": manual_id
    }

    # 1. Parse dynamically passed sensors from form_data
    sensors_json = form_data.get('sensors')
    sensor_configs = {}
    validation_results = {}
    
    if sensors_json:
        try:
            sensors = json.loads(str(sensors_json))
            parser = DatasheetParser()
            
            # Extract configs from each sensor datasheet
            for sensor in sensors:
                s_name = sensor.get("sensor_name", "")
                s_id = sensor.get("sensor_id", "")
                file_key = f"datasheet_{s_id}"
                
                pdf_file: UploadFile = form_data.get(file_key)
                if pdf_file and pdf_file.filename:
                    pdf_bytes = await pdf_file.read()
                    pdf_text = parser.parse_pdf(pdf_bytes)
                    config = parser.extract_sensor_config(s_name, s_id, pdf_text)
                    logger.info(f"📄 Extracted config from PDF for {s_id}")
                else:
                    # No PDF — still use the LLM to estimate from sensor name
                    config = parser.extract_sensor_config_no_pdf(s_name, s_id)
                    logger.info(f"🔧 Generated config from sensor name: {s_id}")
                
                sensor_configs[s_id] = config
                
                # Track validation results
                if "ai_validation" in config:
                    validation_results[s_id] = config["ai_validation"]
            
            # 2. AI Automation Engineer: Cross-validate all sensors together
            try:
                from agents.ai_automation_engineer import AIAutomationEngineerAgent
                ai_engineer = AIAutomationEngineerAgent()
                
                # Determine machine type from name
                machine_type = "industrial_equipment"
                name_lower = name.lower()
                if "pump" in name_lower:
                    machine_type = "pump"
                elif "motor" in name_lower:
                    machine_type = "electric_motor"
                elif "conveyor" in name_lower:
                    machine_type = "conveyor"
                elif "turbine" in name_lower:
                    machine_type = "turbine"
                elif "lathe" in name_lower or "cnc" in name_lower:
                    machine_type = "cnc_machine"
                
                # Cross-validate sensor relationships
                cross_validation = ai_engineer.cross_validate_sensors(sensor_configs, machine_type)
                logger.info(f"🔗 Cross-validation: consistent={cross_validation.get('is_consistent')}, score={cross_validation.get('consistency_score', 0):.2f}")
                
                # Store cross-validation metadata
                validation_results["_cross_validation"] = cross_validation
                
                # 3. Generate AI-powered anomaly patterns for training
                anomaly_patterns = {}
                for anomaly_type in ["machine_fault", "sensor_drift"]:
                    patterns = ai_engineer.generate_anomaly_patterns(sensor_configs, machine_type, anomaly_type)
                    anomaly_patterns[anomaly_type] = patterns
                    logger.info(f"📊 Generated {len(patterns.get('patterns', []))} patterns for {anomaly_type}")
                
                # Upload anomaly patterns to Cloudinary for generate_dataset.py to pick up
                from services import pipeline_storage
                pipeline_storage.upload_pipeline_json(
                    anomaly_patterns, pipeline_storage.anomaly_patterns_public_id(machine_id)
                )

            except Exception as ai_error:
                logger.warning(f"⚠️ AI Engineer workflow partial failure: {ai_error}")
                # Continue without AI enhancements

            # Save to Database SensorConfiguration table (DB is single source of truth)
            db_config = db.query(SensorConfiguration).filter(SensorConfiguration.machine_id == machine_id).first()
            if db_config:
                db_config.config_json = json.dumps(sensor_configs)
                db_config.updated_at = datetime.now().isoformat()
            else:
                db_config = SensorConfiguration(
                    machine_id=machine_id,
                    config_json=json.dumps(sensor_configs),
                    updated_at=datetime.now().isoformat()
                )
                db.add(db_config)
            db.commit()

            # Make the new config visible to this process immediately. Without
            # this the API keeps serving its startup-time cache, so the machine
            # just registered resolves to the default sensor profile.
            from services.sensor_config_loader import sensor_config_loader
            sensor_config_loader.reload(force=True)

            # 4. Trigger async dataset generation and model training.
            # Each stage reads/writes its data via Cloudinary (see services/pipeline_storage.py)
            # instead of local scratch files, so there's nothing left to clean up afterward.
            def run_generation():
                backend_dir = os.path.join(os.path.dirname(__file__), "..")
                try:
                    # sys.executable, not "python": a bare "python" resolves off PATH,
                    # which silently runs these stages in whatever interpreter happens to
                    # be first (e.g. a global install missing cloudinary/sklearn) rather
                    # than the one running the API.
                    py = sys.executable
                    # Generate dataset with AI patterns
                    subprocess.run([py, "scripts/generate_dataset.py", "--machine_id", machine_id, "--use_ai_patterns"], cwd=backend_dir, check=True)
                    # Mandatory Normalization Step (uploads scaler + normalized dataset to cloud)
                    subprocess.run([py, "preprocessing/normalization.py", "--machine_id", machine_id], cwd=backend_dir, check=True)
                    # Train model (uploads model to cloud)
                    subprocess.run([py, "models/train_model.py", "--machine_id", machine_id], cwd=backend_dir, check=True)
                    # Evaluate model performance (Precision, Recall, F1, AUC, etc.)
                    subprocess.run([py, "models/evaluate_model.py", "--machine_id", machine_id, "--model", "both"], cwd=backend_dir, check=True)
                    logger.info(f"✅ Completed ML pipeline for {machine_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to generate datasets/models for {machine_id}: {e}")
            
            # Queued, not awaited. The four stages take minutes; awaiting them
            # here held the HTTP response open that whole time, so the registration
            # form sat disabled behind `loading` and browsers could time out on a
            # request that was actually still succeeding.
            background_tasks.add_task(run_generation)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse sensors/datasheets: {e}")

    # 5. Update Database (only after validation passes)
    existing = db.query(Machine).filter(Machine.machine_id == machine_id).first()
    if existing:
        for key, value in machine_dict.items():
            setattr(existing, key, value)
        db.commit()
        db.refresh(existing)
        return existing
        
    db_machine = Machine(**machine_dict)
    db.add(db_machine)
    db.commit()
    db.refresh(db_machine)
    return db_machine

@router.post("/api/machines/delete/{machine_id}")
async def decommission_machine(machine_id: str, db: Session = Depends(get_db)):
    """Remove a machine from the registry."""
    machine = db.query(Machine).filter(Machine.machine_id == machine_id).first()
    if not machine:
        raise HTTPException(status_code=404, detail="Asset not found")
    
    db.delete(machine)
    db.commit()
    return {"status": "decommissioned", "machine_id": machine_id}
