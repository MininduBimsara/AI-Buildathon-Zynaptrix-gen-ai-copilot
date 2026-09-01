"""
orchestrator_agent.py — Coordinates all agents into a unified pipeline.

NOT WIRED INTO app/main.py — confirmed unreachable from the live FastAPI app
(2026-08, OpenAI->Qwen migration audit). Imports database.neon_vector_store,
which does not exist anywhere in this repo, so `import agents.orchestrator_agent`
raises ModuleNotFoundError today. Left as-is (not migrated, not deleted) so the
original author can reconcile or revive it later. The live LangGraph
(agents/copilot_graph.py) implements its own Sensor/Validation/Diagnostic/
Knowledge/Strategy/Critic nodes inline and does not import this module or its
dependents (machine_health_agent.py, sensor_status_agent.py, strategy_agent.py).
"""
import logging
from typing import Dict, Any

from agents.machine_health_agent import MachineHealthAgent
from agents.sensor_status_agent import SensorStatusAgent
from agents.strategy_agent import StrategyAgent
from agents.knowledge_agent import KnowledgeAgent
from database.neon_vector_store import NeonVectorStore

logger = logging.getLogger(__name__)

class OrchestratorAgent:
    """
    The Boss Agent.
    Orchestrates SensorStatus, Strategy, and Knowledge agents to form
    a comprehensive response to an anomaly event, then logs it to Neon.
    """
    def __init__(self):
        self.health_agent = MachineHealthAgent()
        self.sensor_agent = SensorStatusAgent()
        self.strategy_agent = StrategyAgent()
        # Knowledge agent uses API keys, so it's instantiated per use or kept in orchestrator
        self.knowledge_agent = KnowledgeAgent()
        self.store = NeonVectorStore()
        
    def handle_anomaly(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main pipeline when an anomaly is detected.
        alert_data expected format: {
            "machine_state": "machine_fault",
            "anomaly_score": 12.5,
            "suspect_sensor": "motor_current", # optional
            "recent_readings": {"temperature": 180.5, "motor_current": 7.2, ...} # optional
        }
        """
        logger.info(f"[Orchestrator] Handling anomaly: {alert_data['machine_state']} (Score: {alert_data['anomaly_score']})")
        
        # 1. Check Sensor deviations if readings are provided
        sensor_analysis = {}
        max_dev = 0.0
        suspect = alert_data.get("suspect_sensor")
        if "recent_readings" in alert_data:
            sensor_analysis = self.sensor_agent.analyze_sensor(alert_data["recent_readings"])
            max_dev = sensor_analysis.get("max_deviation_pct", 0.0)
            if not suspect and sensor_analysis.get("suspect_sensor"):
                suspect = sensor_analysis["suspect_sensor"]

        # 2. Get operational strategy
        strategy = self.strategy_agent.recommend_strategy(
            machine_state=alert_data["machine_state"],
            anomaly_score=alert_data["anomaly_score"],
            max_deviation_pct=max_dev
        )

        # 3. Retrieve knowledge base advice (RAG)
        knowledge_result = self.knowledge_agent.query_from_alert({
            "machine_state": alert_data["machine_state"],
            "suspect_sensor": suspect,
            "anomaly_score": alert_data["anomaly_score"]
        })
        advice = knowledge_result["answer"]

        # 4. Log event to vector store
        event_id = None
        try:
            event_id = self.store.log_anomaly_event(
                machine_state=alert_data["machine_state"],
                anomaly_score=float(alert_data["anomaly_score"]),
                suspect_sensor=suspect,
                triggered_by="Monitoring Service Orchestrator",
                agent_advice=f"Strategy: {strategy}\n\nRAG Advice: {advice}"
            )
        except Exception as e:
            logger.error(f"[Orchestrator] Failed to log event to Neon DB: {e}")

        return {
            "event_id": event_id,
            "machine_state": alert_data["machine_state"],
            "anomaly_score": alert_data["anomaly_score"],
            "suspect_sensor": suspect,
            "strategy": strategy,
            "rag_advice": advice,
            "sensor_analysis": sensor_analysis
        }
