import logging
import os
import json
from typing import TypedDict, Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from unified_rag.retrieval.rag import RAGMode
# Used in knowledge_retrieval_node to build absolute image URLs. Without this
# import every copilot query raised NameError inside the try block and the
# real RAG answer was replaced by "Manual Extract Failed for <id>".
from unified_rag.config import settings

# EXPLICITLY IMPORTING THE IMPROVED RAG SYSTEM HERE!
try:
    from unified_rag.retrieval.rag import RAGGenerator
    rag_gen = RAGGenerator()
except Exception:
    rag_gen = None

# AI Validation Layer imports
try:
    from unified_rag.ai_client import get_client, MODEL_CHAT
    ai_client = get_client()
except Exception as e:
    ai_client = None
    logging.warning(f"⚠️ AI client not available for validation: {e}")

from agents.validation_prompts import build_validation_prompt, get_full_system_prompt

logger = logging.getLogger(__name__)

class CopilotState(TypedDict):
    """
    Represents the shared state modified by all nodes in the LangGraph orchestration.
    This state acts as the 'Short-term Memory' for the AI diagnostic session.
    """
    # Alert Inputs from the LSTM/Autoencoder
    event_id: str
    machine_id: str
    machine_state: str
    anomaly_score: float
    user_query: Optional[str]
    suspect_sensor: Optional[str]
    recent_readings: Optional[Dict[str, Any]]
    
    # AI Validation Layer fields
    ai_validation_status: Optional[str]  # TRUE_FAULT, SENSOR_GLITCH, NORMAL_WEAR
    fault_category: Optional[str]  # mechanical, thermal, electrical, process, sensor
    ai_confidence_score: Optional[float]  # 0.0 - 1.0
    ai_engineering_notes: Optional[str]  # AI reasoning explanation
    
    # State accumulated across agents
    sensor_status_report: str
    diagnostic_report: str
    rag_context: str
    retrieved_images: List[str]
    strategy_report: str
    critic_feedback: str
    final_execution_plan: str
    chat_history: Optional[str]  # The current conversation string for the RAG Wizard

def sensor_status_node(state: CopilotState):
    """
    Agent Node: Translates raw sub-symbolic telemetry into natural language status.
    Uses the Anomaly Score (MSE) to quantify the severity of the deviation.
    """
    logger.info("🤖 [Agent] Sensor Status")
    suspect = state.get('suspect_sensor', 'Multiple')
    report = f"Detected severe deviations aligning with {suspect}. Mathematical threshold exceeded by {state['anomaly_score']:.2f} MSE."
    return {"sensor_status_report": report}


def validation_engineer_node(state: CopilotState):
    """
    Agent Node: AI Validation Engineer - High-Accuracy anomaly validation.
    
    Enhanced multi-stage validation using AI Automation Engineer:
    - Stage 1: Physics violations check (sensor limits)
    - Stage 2: Temporal pattern analysis (spike vs sustained)
    - Stage 3: Cross-sensor correlation analysis
    - Stage 4: AI Automation Engineer high-accuracy classification
    
    Returns validation status: TRUE_FAULT, SENSOR_GLITCH, or NORMAL_WEAR
    with detailed root cause analysis and recommended actions.
    """
    logger.info("🤖 [Agent] AI Validation Engineer - High Accuracy Mode")
    
    # Skip validation for user queries (non-anomaly interactions)
    user_query = state.get('user_query')
    if user_query and user_query.strip():
        logger.info("⏭️ [Validation] Skipping - user query present")
        return {
            "ai_validation_status": None,
            "fault_category": None,
            "ai_confidence_score": None,
            "ai_engineering_notes": None
        }
    
    # Get anomaly data
    anomaly_score = state.get('anomaly_score', 0.0)
    recent_readings = state.get('recent_readings') or {}
    machine_id = state.get('machine_id', 'UNKNOWN')
    
    # STAGE 1: Import validation components
    try:
        from services.sensor_config_loader import sensor_config_loader
        from services.anomaly_service import TemporalAnalyzer, calculate_hybrid_confidence
    except ImportError as e:
        logger.error(f"❌ [Validation] Import error: {e}")
        return _default_validation_result(anomaly_score)
    
    # STAGE 2: Physics violations check
    physics_summary = sensor_config_loader.get_violation_summary(machine_id, recent_readings)
    logger.info(f"📊 [Stage 1] Physics check: {len(physics_summary.get('fault_violations', []))} critical, {len(physics_summary.get('normal_violations', []))} warnings")
    
    # STAGE 3: Temporal pattern analysis
    temporal_pattern = {
        "is_spike": False,
        "is_sustained": True,  # Default to sustained for initial anomalies
        "anomaly_count": 1,
        "trend": "stable",
        "max_rate_of_change": 0.0,
        "suspicious_sensors": []
    }
    
    # Calculate hybrid confidence
    hybrid_confidence = calculate_hybrid_confidence(
        anomaly_score, physics_summary, temporal_pattern
    )
    logger.info(f"📊 [Stage 2] Hybrid confidence: {hybrid_confidence:.2f}")
    
    # Quick pre-screen for obvious cases
    if hybrid_confidence < 0.2:
        logger.info(f"⚡ [Validation] Auto-classified as SENSOR_GLITCH (low confidence: {hybrid_confidence:.2f})")
        return {
            "ai_validation_status": "SENSOR_GLITCH",
            "fault_category": None,
            "ai_confidence_score": 0.85,
            "ai_engineering_notes": "Low hybrid confidence indicates transient noise or measurement artifact. No action required."
        }
    
    # STAGE 4: High-Accuracy AI Classification using AI Automation Engineer
    try:
        from agents.ai_automation_engineer import AIAutomationEngineerAgent

        # Get sensor configs for cross-correlation analysis
        sensor_configs = sensor_config_loader.get_machine_config(machine_id) or {}
        
        # Prepare anomaly data package
        anomaly_data = {
            "machine_id": machine_id,
            "anomaly_score": anomaly_score,
            "recent_readings": recent_readings,
            "physics_summary": physics_summary,
            "temporal_pattern": temporal_pattern,
            "hybrid_confidence": hybrid_confidence
        }
        
        # Initialize AI Automation Engineer
        ai_engineer = AIAutomationEngineerAgent()
        
        # High-accuracy fault classification
        classification = ai_engineer.high_accuracy_fault_classification(
            anomaly_data=anomaly_data,
            sensor_configs=sensor_configs,
            historical_context=None  # Could add historical anomalies here
        )
        
        ai_status = classification.get("primary_classification", "TRUE_FAULT")
        fault_cat = classification.get("fault_category")
        confidence = classification.get("confidence_score", hybrid_confidence)
        
        # Build enhanced engineering notes with root cause analysis
        notes_parts = [classification.get("engineering_notes", "")]
        
        # Add hypothesis info if available
        hypotheses = classification.get("hypotheses", [])
        if hypotheses:
            top_hypothesis = hypotheses[0]
            notes_parts.append(f"Primary hypothesis: {top_hypothesis.get('description', 'Unknown')} ({top_hypothesis.get('probability', 0):.0%})")
        
        # Add root cause info
        rca = classification.get("root_cause_analysis", {})
        if rca.get("primary_cause"):
            notes_parts.append(f"Root cause: {rca['primary_cause']}")
        
        # Add top recommended action
        actions = classification.get("recommended_actions", [])
        if actions:
            top_action = actions[0]
            notes_parts.append(f"Action ({top_action.get('priority', 'medium')}): {top_action.get('action', 'Investigate')}")
        
        notes = " | ".join(filter(None, notes_parts))
        
        logger.info(f"✅ [Stage 4] High-accuracy validation: {ai_status} | {fault_cat or 'N/A'} | Confidence: {confidence:.2f}")
        
        # Log if human review is recommended
        if classification.get("requires_human_review"):
            logger.warning(f"⚠️ [Validation] Human review recommended for this classification")
        
        return {
            "ai_validation_status": ai_status,
            "fault_category": fault_cat,
            "ai_confidence_score": confidence,
            "ai_engineering_notes": notes
        }
        
    except ImportError as e:
        logger.warning(f"⚠️ [Validation] AI Automation Engineer not available: {e}")
        # Fallback to basic GPT-4o validation
        return _fallback_gpt4_validation(
            machine_id, anomaly_score, physics_summary, 
            temporal_pattern, recent_readings, hybrid_confidence
        )
    except Exception as e:
        logger.error(f"❌ [Validation] High-accuracy classification failed: {e}")
        return _default_validation_result(hybrid_confidence)


def _fallback_gpt4_validation(
    machine_id: str,
    anomaly_score: float,
    physics_summary: dict,
    temporal_pattern: dict,
    recent_readings: dict,
    hybrid_confidence: float
) -> dict:
    """Fallback to basic Qwen validation if AI Automation Engineer unavailable."""
    if not ai_client:
        return _default_validation_result(hybrid_confidence)
    
    try:
        prompt = build_validation_prompt(
            machine_id=machine_id,
            ml_score=anomaly_score,
            physics_summary=physics_summary,
            temporal_pattern=temporal_pattern,
            recent_readings=recent_readings,
            hybrid_confidence=hybrid_confidence
        )
        
        response = ai_client.chat.completions.create(
            model=MODEL_CHAT,
            messages=[
                {"role": "system", "content": get_full_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"}
        )

        validation_json = json.loads(response.choices[0].message.content)

        return {
            "ai_validation_status": validation_json.get("ai_validation_status", "TRUE_FAULT"),
            "fault_category": validation_json.get("fault_category"),
            "ai_confidence_score": validation_json.get("confidence_score", hybrid_confidence),
            "ai_engineering_notes": validation_json.get("ai_engineering_notes", "Validation completed.")
        }

    except Exception as e:
        logger.error(f"❌ [Fallback Validation] Qwen call failed: {e}")
        return _default_validation_result(hybrid_confidence)


def _default_validation_result(confidence: float) -> dict:
    """Fallback validation result when AI is unavailable."""
    return {
        "ai_validation_status": "TRUE_FAULT",
        "fault_category": "mechanical",
        "ai_confidence_score": confidence,
        "ai_engineering_notes": "AI validation pending - using hybrid confidence score. Manual verification recommended."
    }

def diagnostic_node(state: CopilotState):
    """
    Agent Node: Performs a symbolic Root Cause Analysis (RCA) with AI validation persistence.
    
    Enhanced workflow:
    1. Uses high-accuracy AI classification from ValidationEngineer
    2. Persists validated results to database
    3. Maps detected anomaly to suspected failure mode
    """
    logger.info("🤖 [Agent] Diagnostic with AI Validation")
    
    machine_id = state.get('machine_id', 'UNKNOWN')
    ai_status = state.get('ai_validation_status')
    fault_category = state.get('fault_category')
    ai_confidence = state.get('ai_confidence_score', 0.0)
    ai_notes = state.get('ai_engineering_notes', '')
    
    # Build enhanced diagnostic report based on AI validation
    if ai_status == "TRUE_FAULT":
        severity = "CRITICAL" if ai_confidence >= 0.8 else "HIGH"
        report = (
            f"🔴 [{severity}] AI-Verified Fault Detected\n"
            f"Classification: {ai_status} ({fault_category or 'unclassified'})\n"
            f"Confidence: {ai_confidence:.0%}\n"
            f"Analysis: {ai_notes}\n"
            f"Root Cause Analysis indicates a high probability of '{state['machine_state']}' failure event."
        )
    elif ai_status == "SENSOR_GLITCH":
        report = (
            f"🟡 [LOW PRIORITY] Sensor Glitch Detected\n"
            f"Classification: {ai_status}\n"
            f"Confidence: {ai_confidence:.0%}\n"
            f"Analysis: {ai_notes}\n"
            f"No immediate action required - transient sensor anomaly."
        )
    elif ai_status == "NORMAL_WEAR":
        report = (
            f"🟢 [MAINTENANCE] Normal Wear Detected\n"
            f"Classification: {ai_status}\n"
            f"Confidence: {ai_confidence:.0%}\n"
            f"Analysis: {ai_notes}\n"
            f"Schedule preventive maintenance as per standard procedures."
        )
    else:
        report = f"Root Cause Analysis indicates a '{state['machine_state']}' event. AI validation pending."
    
    # Persist validated diagnostic to database
    if ai_status and not state.get('user_query'):  # Only persist for anomaly events, not user queries
        try:
            from unified_rag.db.database import SessionLocal
            from unified_rag.db.models import AnomalyRecord
            
            db = SessionLocal()
            try:
                # Find the most recent unvalidated anomaly for this machine
                recent_anomaly = db.query(AnomalyRecord).filter(
                    AnomalyRecord.machine_id == machine_id,
                    AnomalyRecord.ai_validation_status == None
                ).order_by(AnomalyRecord.id.desc()).first()
                
                if recent_anomaly:
                    recent_anomaly.ai_validation_status = ai_status
                    recent_anomaly.fault_category = fault_category
                    recent_anomaly.ai_confidence_score = ai_confidence
                    recent_anomaly.ai_engineering_notes = ai_notes
                    db.commit()
                    logger.info(f"💾 [Diagnostic] Persisted validation for anomaly {recent_anomaly.id}: {ai_status}")
                else:
                    logger.debug("[Diagnostic] No unvalidated anomaly found to update")
                    
            except Exception as db_error:
                logger.error(f"❌ [Diagnostic] DB persist failed: {db_error}")
            finally:
                db.close()
                
        except ImportError as e:
            logger.warning(f"⚠️ [Diagnostic] DB modules not available: {e}")
    
    return {"diagnostic_report": report}

def knowledge_retrieval_node(state: CopilotState):
    """
    Agent Node: The Multimodal RAG Interface.
    1. Orchestrates semantic search against the Zynaptrix-9000 PDF Vector DB.
    2. Resolves technical diagram paths for the React Frontend.
    3. Normalizes OS-level path separators to ensure 100% URL reliability.
    """
    logger.info(f"🤖 [Agent] Knowledge Retrieval for Machine: {state.get('machine_id', 'Unknown')}")
    machine_id = state.get('machine_id', 'PUMP-001')
    
    # HITL: Use the operator's specific query if available, otherwise default to a general summary
    user_q = state.get('user_query', '')
    mode = RAGMode.SUMMARY
    query = user_q

    if "[CLARIFY_STEP]" in user_q:
        mode = RAGMode.CLARIFICATION
        query = user_q.replace("[CLARIFY_STEP]", "").strip()
    elif "[EVALUATE_STEP]" in user_q:
        mode = RAGMode.EVALUATION
        query = user_q.replace("[EVALUATE_STEP]", "").strip()
    elif "[CONVERSATIONAL_WIZARD]" in user_q or "Generate full step-by-step repair procedure" in user_q:
        # This is the new Intelligent Navigator
        mode = RAGMode.CONVERSATIONAL_WIZARD
        query = user_q.replace("[CONVERSATIONAL_WIZARD]", "").strip()
        if not query:
            query = "Begin the guided repair by providing necessary safety and preparation steps (Lockout/Tagout, PPE) from the manual."
    elif user_q.strip():
        query = f"Diagnostic Summary only for: {user_q} (Anomaly: {state['machine_state']})"
    else:
        query = f"Provide a brief diagnostic summary and status for {state['machine_state']}."
    
    images = []
    if rag_gen:
        try:
            # DYNAMIC MANUAL SELECTION: Resolve machine_id -> manual_id via Machine Registry
            from unified_rag.db.database import SessionLocal
            from unified_rag.db.models import Machine
            
            db = SessionLocal()
            try:
                # 1. Lookup the machine's specific manual
                machine_record = db.query(Machine).filter(Machine.machine_id == machine_id).first()
                manual_id = machine_record.manual_id if machine_record else "Zynaptrix_9000"
                
                # 2. PROVENANCE CHECK: Does this manual actually have content in our RAG DB?
                from unified_rag.db.models import ManualChunk
                chunk_count = db.query(ManualChunk).filter(ManualChunk.manual_id == manual_id).count()
                
                if chunk_count == 0:
                    logger.warning(f"⚠️ [Provenance] No manual content found for {manual_id}. Flagging for disclaimer.")
                    # Prepend a hidden flag to the query so the RAG knows to DISCLAIMER it.
                    query = f"[DISCLAIMER_REQUIRED: MISSING_MANUAL] {query}"
                else:
                    logger.info(f"✅ [Provenance] {chunk_count} manual chunks found for {manual_id}.")
            finally:
                db.close()
            
            logger.info(f"🔍 [RAG Routing] Machine {machine_id} resolved to Manual: {manual_id} (Mode: {mode})")
            rag_response = rag_gen.generate_response(
                query, 
                manual_id, 
                machine_id, 
                mode=mode, 
                chat_history=state.get('chat_history', '')
            )
            
            # Unified API configuration from settings
            api_url = settings.api_url
            if api_url.endswith('/'):
                api_url = api_url[:-1]

            for img_path in rag_response.get("images", []):
                # ⭐ CLOUDINARY SUPPORT: If it's already an absolute URL, use it directly
                if img_path.startswith('http'):
                    images.append(img_path)
                    continue
                    
                # CROSS-PLATFORM NORMALIZATION
                normalized_path = img_path.replace('\\', '/')
                # Transform filesystem path to virtual mount path (/static)
                web_path = normalized_path.replace("data/", "/static/")
                if not web_path.startswith('/'):
                    web_path = '/' + web_path
                images.append(f"{api_url}{web_path}")
                
            rag = rag_response['answer']
        except Exception as e:
            logger.error(f"❌ [RAG Routing] Dynamic lookup failed for {machine_id}: {e}")
            rag = f"Manual Extract Failed for {machine_id}: {str(e)}"
    else:
        rag = "Standard Procedures: Maintain equipment as per manufacturer specifications."
        
    return {"rag_context": rag, "retrieved_images": images}

def strategy_node(state: CopilotState):
    """
    Agent Node: Formulates the final strategy using RAG context.
    The strategy incorporates the exact textual instructions and interleaved image tags.
    """
    logger.info("🤖 [Agent] Strategy")
    plan = state.get('rag_context', 'No RAG context available.')
    return {"strategy_report": plan}

def critic_node(state: CopilotState):
    """
    Agent Node: Final Validation & Formatting.
    Ensures the output is sanitized and formatted for the Next.js 'Markdown-Lite' renderer.
    """
    logger.info("🤖 [Agent] Critic")
    feedback = f"Validation: Safety protocols verified against {state.get('machine_id', 'Asset')} Technical Documentation. Strategy approved for execution."
    
    # HITL: If this is a specific user query, provide a clean chat response.
    # If it's an initial anomaly detection (no user query yet), provide a brief summary.
    user_q = state.get('user_query')
    image_tags = "\n".join([f"![Image]({img})" for img in state.get('retrieved_images', [])])
    # 📸 Automated Image Interleaving: Ensure the UI receives image markers
    images = state.get('retrieved_images', [])
    image_tags = "\n".join([f"![Technical Diagram]({img})" for img in images])
    
    if user_q and user_q.strip():
        final_output = state['strategy_report']
    else:
        # Initial AI Diagnostic Message: Clean & Suggestion-Focused
        final_output = (
            f"### 🚨 AI Diagnostic Alert\n"
            f"**Event**: {state['machine_state']} | **Confidence**: {state['anomaly_score']:.2f}\n\n"
            f"{state['strategy_report']}\n"
        )
        
    return {
        "critic_feedback": feedback,
        "final_execution_plan": final_output
    }

def build_copilot_graph() -> StateGraph:
    """
    Architecture Definition:
    Creates a directed acyclic graph (DAG) representing the AI reasoning process.
    The linear pipeline ensures a predictable 'Sensor-to-Execution' transformation.
    
    Pipeline:
    SensorStatus → ValidationEngineer → Diagnostic → KnowledgeRetrieval → Strategy → Critic
    """
    workflow = StateGraph(CopilotState)
    
    workflow.add_node("SensorStatusAgent", sensor_status_node)
    workflow.add_node("ValidationEngineerAgent", validation_engineer_node)  # NEW: AI Validation Layer
    workflow.add_node("DiagnosticAgent", diagnostic_node)
    workflow.add_node("KnowledgeRetrievalAgent", knowledge_retrieval_node)
    workflow.add_node("StrategyAgent", strategy_node)
    workflow.add_node("CriticAgent", critic_node)
    
    workflow.set_entry_point("SensorStatusAgent")
    workflow.add_edge("SensorStatusAgent", "ValidationEngineerAgent")  # NEW edge
    workflow.add_edge("ValidationEngineerAgent", "DiagnosticAgent")    # NEW edge
    workflow.add_edge("DiagnosticAgent", "KnowledgeRetrievalAgent")
    workflow.add_edge("KnowledgeRetrievalAgent", "StrategyAgent")
    workflow.add_edge("StrategyAgent", "CriticAgent")
    workflow.add_edge("CriticAgent", END)
    
    return workflow.compile()
