import logging
import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import desc
from unified_rag.ai_client import get_client, MODEL_CHAT, MODEL_CHAT_LIGHT

from unified_rag.config import settings
from unified_rag.retrieval.rag import RAGGenerator, RAGMode
from unified_rag.db.database import get_db
from unified_rag.db.models import AssistantSession, AssistantMessage, Machine

from app.constants import SYSTEM_ONBOARDING_CONTEXT, SYSTEM_GUIDE_STEPS

router = APIRouter(prefix="/api/assistant", tags=["Central Assistant"])
logger = logging.getLogger(__name__)

# Models
class AssistantQuery(BaseModel):
    query: str
    session_id: Optional[int] = None
    machine_id: Optional[str] = None

class AssistantSessionResponse(BaseModel):
    id: int
    machine_id: Optional[str]
    title: str
    timestamp: str

class AssistantMessageView(BaseModel):
    role: str
    content: str
    type: str
    step_data: Optional[Dict]
    images: List[str]
    timestamp: str

# Helper Functions
def detect_guide_topic(query: str) -> str | None:
    """Detect if query is asking about system usage and return the topic key."""
    q = query.lower()
    if any(x in q for x in ["ingest", "upload", "manual", "pdf", "add manual", "import"]):
        return "ingest"
    if any(x in q for x in ["register machine", "add machine", "create machine", "add sensor", "add sensors", "datasheet", "train model"]):
        return "register"
    if any(x in q for x in ["simulator", "start simulation", "simulate", "telemetry stream"]):
        return "simulator"
    if any(x in q for x in ["how to use assistant", "use this bot", "rag manual", "assistant help"]):
        return "assistant"
    if any(x in q for x in ["framework", "architecture", "system works", "gen ai", "how this works", "what is this"]):
        return "framework"
    return None

def generate_session_title(query: str) -> str:
    """Generate a 3-4 word title for a new session based on the first query."""
    client = get_client()
    try:
        res = client.chat.completions.create(
            model=MODEL_CHAT_LIGHT,
            messages=[{"role": "user", "content": f"Generate a very short title (max 4 words) for a chat that starts with this question: '{query}'. Provide ONLY the title."}],
            max_tokens=20
        )
        return res.choices[0].message.content.strip().strip('"')
    except Exception as e:
        logger.warning(f"Failed to generate title: {e}")
        return "New Assistant Inquiry"

def perform_web_search(query: str) -> str:
    """Simulated web search for industrial knowledge."""
    client = get_client()
    logger.info(f"🌐 Performing simulated web search for: {query}")
    try:
        search_prompt = (
            f"You are a specialized Web Search Engine for Industrial IoT and Maintenance. "
            f"Provide a summary of the latest information regarding: '{query}'. "
            f"Include real-world trends, safety standards (like ISO/OSHA), and industry news if applicable."
        )
        res = client.chat.completions.create(
            model=MODEL_CHAT_LIGHT,
            messages=[{"role": "user", "content": search_prompt}],
            max_tokens=600
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"Search service unavailable: {str(e)}"

# ───── Endpoints ──────────────────────────────────────────────────────────────

@router.get("/sessions", response_model=List[AssistantSessionResponse])
async def get_assistant_sessions(db: Session = Depends(get_db)):
    sessions = db.query(AssistantSession).order_by(desc(AssistantSession.updated_at)).all()
    return [{
        "id": s.id,
        "machine_id": s.machine_id,
        "title": s.title,
        "timestamp": s.updated_at
    } for s in sessions]

@router.delete("/sessions/{session_id}")
async def delete_assistant_session(session_id: int, db: Session = Depends(get_db)):
    db.query(AssistantMessage).filter(AssistantMessage.session_id == session_id).delete()
    db.query(AssistantSession).filter(AssistantSession.id == session_id).delete()
    db.commit()
    return {"status": "success"}

@router.get("/sessions/{session_id}/history", response_model=List[AssistantMessageView])
async def get_session_history(session_id: int, db: Session = Depends(get_db)):
    messages = db.query(AssistantMessage).filter(AssistantMessage.session_id == session_id).order_by(AssistantMessage.timestamp).all()
    return [{
        "role": m.role,
        "content": m.content,
        "type": m.type,
        "step_data": json.loads(m.step_data) if m.step_data else None,
        "images": json.loads(m.images) if m.images else [],
        "timestamp": m.timestamp
    } for m in messages]

@router.get("/sessions/{session_id}/report")
async def generate_report(session_id: int, db: Session = Depends(get_db)):
    """Generate structured diagnostic report from session using AI."""
    session = db.query(AssistantSession).filter(AssistantSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = db.query(AssistantMessage).filter(AssistantMessage.session_id == session_id).order_by(AssistantMessage.timestamp).all()
    if not messages:
        raise HTTPException(status_code=400, detail="No messages in session")
    
    conversation = []
    all_images = []
    for m in messages:
        conversation.append(f"{m.role.upper()}: {m.content}")
        if m.images:
            try:
                all_images.extend(json.loads(m.images))
            except: pass
    
    conversation_text = "\n\n".join(conversation)
    client = get_client()
    
    system_prompt = """You are a technical documentation expert. Extract key information from this diagnostic conversation and structure it as a professional maintenance report in JSON format.
    Output must be a valid JSON object with the following keys: problem, diagnosis, solution_steps (a list of strings)."""

    try:
        response = client.chat.completions.create(
            model=MODEL_CHAT_LIGHT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract report data:\n\n{conversation_text}"}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        extracted = json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Report AI failed: {e}")
        extracted = {"problem": "N/A", "diagnosis": "N/A", "solution_steps": []}

    return {
        "sessionId": session.id,
        "machineId": session.machine_id,
        "problemDescription": extracted.get("problem", "N/A"),
        "diagnosis": extracted.get("diagnosis", "N/A"),
        "solutionSteps": extracted.get("solution_steps", []),
        "images": [{"url": u, "caption": f"Ref {i+1}"} for i, u in enumerate(list(set(all_images)))],
        "timestamp": session.created_at
    }

@router.post("")
async def system_assistant(req: AssistantQuery, db: Session = Depends(get_db)):
    """Main stateful interaction endpoint for the Central Assistant."""
    client = get_client()
    active_session_id = req.session_id
    active_machine_id = req.machine_id

    # Session Management
    if active_session_id:
        session = db.query(AssistantSession).filter(AssistantSession.id == active_session_id).first()
        if session:
            if req.machine_id: session.machine_id = req.machine_id
            active_machine_id = active_machine_id or session.machine_id
    else:
        new_session = AssistantSession(
            machine_id=active_machine_id,
            title=generate_session_title(req.query),
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            updated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
        active_session_id = new_session.id
    
    # Save User Msg
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db.add(AssistantMessage(session_id=active_session_id, role="user", content=req.query, timestamp=now))
    db.commit()

    # Context Fetch
    history = db.query(AssistantMessage).filter(AssistantMessage.session_id == active_session_id).order_by(AssistantMessage.timestamp.desc()).offset(1).limit(10).all()
    chat_context = [{"role": ("assistant" if m.role == "agent" else m.role), "content": m.content} for m in reversed(history)]

    # Intent Classification
    guide_topic = detect_guide_topic(req.query)
    
    if active_machine_id:
        # If the user explicitly selected a machine/manual context, FORCE the RAG pipeline.
        intent = "RAG"
    elif guide_topic:
        intent = "GUIDE"
    else:
        intent_prompt = (
            "Classify the user intent into one of these categories:\n"
            "- ONBOARDING: Help with using THIS assistant or the framework software.\n"
            "- RAG: Technical questions about industrial equipment, maintenance procedures, cleaning, startup/shutdown, or technical specs.\n"
            "- SEARCH: Industry news, general trends, or external standards.\n"
            "- CHAT: Greetings, small talk, or general non-technical questions.\n"
            "Respond with ONLY the category name."
        )
        res = client.chat.completions.create(
            model=MODEL_CHAT_LIGHT,
            messages=[{"role": "system", "content": "You are a technical intent router."}] + chat_context + [{"role": "user", "content": intent_prompt}],
            max_tokens=10, temperature=0
        )
        intent = res.choices[0].message.content.strip().upper()

    # Execution
    images = []
    context_source = "AI"
    if intent == "GUIDE":
        steps = SYSTEM_GUIDE_STEPS.get(guide_topic, [])
        final_answer = "\n\n".join([f"**{i}. {s['title']}**\n{s['detail']}" for i, s in enumerate(steps, 1)])
        context_source = "System Guide"
    elif intent == "ONBOARDING":
        res = client.chat.completions.create(
            model=MODEL_CHAT,
            messages=[{"role": "system", "content": f"System Docs: {SYSTEM_ONBOARDING_CONTEXT}"}] + chat_context + [{"role": "user", "content": req.query}]
        )
        final_answer = res.choices[0].message.content
        context_source = "Onboarding"
    elif intent == "RAG" and active_machine_id:
        rag_gen = RAGGenerator()
        
        # 1. Resolve Manual ID from machine registry
        # We handle case where active_machine_id might be a string-int ID from a legacy UI state
        machine_record = db.query(Machine).filter(Machine.machine_id == str(active_machine_id)).first()
        if not machine_record and str(active_machine_id).isdigit():
            machine_record = db.query(Machine).filter(Machine.id == int(active_machine_id)).first()
            
        manual_id = machine_record.manual_id if machine_record else "Zynaptrix_9000"
        
        # 2. Build history string for RAG engine
        chat_history_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in chat_context])
        
        mode = RAGMode.CONVERSATIONAL_WIZARD if "step" in req.query.lower() else RAGMode.SUMMARY
        rag_res = rag_gen.generate_response(
            req.query, 
            manual_id, 
            active_machine_id, 
            mode=mode,
            chat_history=chat_history_str
        )
        
        # 📸 Normalize image paths to full Web URLs
        raw_images = rag_res.get('images', [])
        images = []
        # Unified API configuration from settings
        api_url = settings.api_url
        if api_url.endswith('/'):
            api_url = api_url[:-1]

        for img_path in raw_images:
            if img_path.startswith('http'):
                images.append(img_path)
                continue
                
            normalized_path = img_path.replace('\\', '/')
            web_path = normalized_path.replace("data/", "/static/")
            if not web_path.startswith('/'):
                web_path = '/' + web_path
            images.append(f"{api_url}{web_path}")

        # ✅ CRITICAL: Use RAG answer DIRECTLY to preserve image tags.
        # Redundant second-pass LLM calls strip away [IMAGE_N] markers.
        final_answer = rag_res.get('answer', 'No response generated.')
        context_source = f"Manual ({manual_id})"
    elif intent == "SEARCH":
        search_data = perform_web_search(req.query)
        res = client.chat.completions.create(
            model=MODEL_CHAT,
            messages=[{"role": "system", "content": f"Search Results: {search_data}"}] + chat_context + [{"role": "user", "content": req.query}]
        )
        final_answer = res.choices[0].message.content
        context_source = "Web Search"
    else:
        res = client.chat.completions.create(model=MODEL_CHAT_LIGHT, messages=chat_context + [{"role": "user", "content": req.query}])
        final_answer = res.choices[0].message.content

    # Persist agent response
    agent_msg = AssistantMessage(
        session_id=active_session_id, role="agent", content=final_answer, 
        type="text", images=json.dumps(images) if images else None, timestamp=now
    )
    db.add(agent_msg)
    # Update session ts
    session_obj = db.query(AssistantSession).filter(AssistantSession.id == active_session_id).first()
    if session_obj: session_obj.updated_at = now
    db.commit()

    return {
        "role": "agent", "content": final_answer, "session_id": active_session_id,
        "images": images, "context_source": context_source, "timestamp": now
    }
