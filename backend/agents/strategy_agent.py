"""
strategy_agent.py — Prescribes operational strategy based on anomaly severity.

NOT WIRED INTO app/main.py — confirmed unreachable from the live FastAPI app
(2026-08, OpenAI->Qwen migration audit). Only imported by the dead
agents/orchestrator_agent.py. The live LangGraph (agents/copilot_graph.py) has
its own inline strategy_node instead. Left as-is.
"""
from typing import Dict, Any

class StrategyAgent:
    def recommend_strategy(self, machine_state: str, anomaly_score: float, max_deviation_pct: float) -> str:
        """Determines if the machine should be stopped, scheduled for maintenance, or watched."""
        state_lower = machine_state.lower()
        
        if state_lower == "idle":
            return "No action required. Machine is currently idle."
            
        if "fault" in state_lower or max_deviation_pct > 25.0:
            return "CRITICAL: Immediate machine shutdown recommended to prevent mechanical damage."
            
        if "drift" in state_lower or "freeze" in state_lower:
            return "WARNING: Schedule maintenance at end of current shift. Sensor recalibration or replacement needed."
            
        if anomaly_score > 0.5:
            return "NOTICE: Monitor machine closely. Anomaly score elevated but within operational limits."
            
        return "Normal operation."
