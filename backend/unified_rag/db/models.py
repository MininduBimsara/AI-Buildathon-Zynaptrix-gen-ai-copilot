from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey, Float
from pgvector.sqlalchemy import Vector
from unified_rag.db.database import Base

class ManualChunk(Base):
    __tablename__ = "manual_chunks"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    manual_id = Column(String, index=True, nullable=False)
    type = Column(String, nullable=False) # 'text', 'image', 'table'
    content = Column(Text, nullable=True) # Text content or structured table string
    embedding = Column(Vector, nullable=False) # Dimension 1536 (OpenAI text) or 512 (CLIP image)
    page = Column(Integer, nullable=True)
    path = Column(String, nullable=True) # Path to the extracted image file

class Machine(Base):
    __tablename__ = "machines"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    location = Column(String, nullable=True)
    manual_id = Column(String, nullable=False) # Maps to ManualChunk.manual_id

class AnomalyRecord(Base):
    __tablename__ = "anomaly_records"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True, nullable=False)
    timestamp = Column(String, nullable=False)
    type = Column(String, nullable=False) # e.g. 'PUMP_FAULT'
    score = Column(Float, nullable=False) # MSE reconstruction error
    sensor_data = Column(Text, nullable=True) # JSON string of readings
    resolved = Column(Boolean, default=False) # Manual HITL sign-off
    
    # Alert Metadata (moved from local JSONL)
    severity = Column(String, nullable=True) # WARNING, HIGH
    suspect_sensor = Column(String, nullable=True)
    threshold = Column(Float, nullable=True)
    
    # AI Validation Layer fields (added 2026-04-06)
    ai_validation_status = Column(String, nullable=True)  # TRUE_FAULT, SENSOR_GLITCH, NORMAL_WEAR
    fault_category = Column(String, nullable=True)  # mechanical, thermal, electrical, process, sensor
    ai_confidence_score = Column(Float, nullable=True)  # 0.0 - 1.0
    ai_engineering_notes = Column(Text, nullable=True)  # AI reasoning explanation

class ChatMessage(Base):
    __tablename__ = "chat_history"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    anomaly_id = Column(Integer, ForeignKey("anomaly_records.id"), nullable=True) # NULL for general chat
    role = Column(String, nullable=False) # 'agent' | 'user'
    content = Column(Text, nullable=False)
    timestamp = Column(String, nullable=False)
    images = Column(Text, nullable=True) # JSON list of URLs for agent responses
    # Use 'message_metadata' in both Python and the DB to avoid conflict with SQLAlchemy's reserved 'metadata' attribute.
    message_metadata = Column(Text, nullable=True) # JSON: procedure state, task completion, etc.

class InteractionMemory(Base):
    """
    Vectorized 'Historical Knowledge' derived from resolved incidents.
    This allows the RAG engine to prioritize previous successful fixes.
    """
    __tablename__ = "interaction_memory"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True, nullable=False)
    manual_id = Column(String, nullable=False) # Origin manual
    summary = Column(Text, nullable=False) # Actionable summary (steps performed)
    operator_fix = Column(Text, nullable=True) # Final operator input
    embedding = Column(Vector, nullable=False) # 1536 OpenAI
    timestamp = Column(String, nullable=False)

class AssistantSession(Base):
    __tablename__ = "assistant_sessions"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True, nullable=True) # Context machine
    title = Column(String, nullable=False)
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)

class AssistantMessage(Base):
    __tablename__ = "assistant_messages"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("assistant_sessions.id"), nullable=False)
    role = Column(String, nullable=False) # 'agent' | 'user'
    content = Column(Text, nullable=False)
    type = Column(String, default='text') # 'text', 'wizard_step', etc.
    step_data = Column(Text, nullable=True) # JSON string
    images = Column(Text, nullable=True) # JSON list
    timestamp = Column(String, nullable=False)


class Manual(Base):
    """Stores metadata for source PDF manuals stored in Cloudinary."""
    __tablename__ = "manuals"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    manual_id = Column(String, unique=True, index=True, nullable=False)
    filename = Column(String, nullable=False)
    url = Column(String, nullable=False) # Cloudinary URL
    created_at = Column(String, nullable=True)


class SensorConfiguration(Base):
    """Stores machine sensor physics limits (replaces legacy sensor_configs.json)."""
    __tablename__ = "sensor_configurations"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True, nullable=False)
    config_json = Column(Text, nullable=False) # Full JSON dump of the sensor limits
    updated_at = Column(String, nullable=True)


class AnomalyThreshold(Base):
    """Stores MSE thresholds from training (replaces legacy thresholds.json)."""
    __tablename__ = "anomaly_thresholds"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True, nullable=False)
    threshold_type = Column(String, default="dense") # 'dense' or 'lstm'
    value = Column(Float, nullable=False)
    updated_at = Column(String, nullable=True)


class MachineAsset(Base):
    """Tracks Cloudinary URLs for trained AI assets (models and scalers)."""
    __tablename__ = "machine_assets"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True, nullable=False)
    asset_type = Column(String, nullable=False) # 'model_dense', 'model_lstm', 'scaler'
    url = Column(String, nullable=False) # Cloudinary URL
    updated_at = Column(String, nullable=True)


class MachineEvaluation(Base):
    """Stores model evaluation metrics + Cloudinary plot URLs (replaces legacy data/processed/evaluation/*.json)."""
    __tablename__ = "machine_evaluations"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True, nullable=False)
    model_type = Column(String, nullable=False)   # 'dense' | 'lstm'
    metrics_json = Column(Text, nullable=False)
    plot_urls_json = Column(Text, nullable=False)  # {"roc_curve": url, "mse_distribution": url, ...}
    evaluated_at = Column(String, nullable=False)

