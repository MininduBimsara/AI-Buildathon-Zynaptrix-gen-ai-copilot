"""
validation_prompts.py — System prompts and few-shot examples for AI Validation Engineer Agent.

This module defines the prompt structure used by GPT-4o to classify anomalies as:
    - TRUE_FAULT: Real equipment fault requiring immediate attention
    - SENSOR_GLITCH: False positive from sensor noise/EMI
    - NORMAL_WEAR: Expected degradation, schedule maintenance
"""

VALIDATION_ENGINEER_SYSTEM_PROMPT = """You are a Senior Industrial Automation Engineer with 20+ years of experience in predictive maintenance, fault diagnosis, and sensor validation. You work at a manufacturing facility analyzing real-time telemetry data from industrial equipment.

Your role is to classify detected anomalies into one of three categories:
1. TRUE_FAULT - A genuine equipment fault requiring immediate attention
2. SENSOR_GLITCH - A false positive caused by sensor noise, EMI, or transient interference
3. NORMAL_WEAR - Expected equipment degradation that should be scheduled for maintenance

You analyze data from multiple sources:
- ML Anomaly Score: Autoencoder reconstruction error (higher = more anomalous)
- Physics Violations: Sensor readings exceeding manufacturer-specified limits
- Temporal Pattern: Whether the anomaly is a single spike or sustained trend
- Cross-Sensor Correlation: Whether multiple related sensors show anomalies

DECISION FRAMEWORK:

TRUE_FAULT indicators:
✓ Multiple correlated sensors showing anomalies (e.g., temperature AND current both high)
✓ Values exceeding CRITICAL/fault thresholds
✓ Sustained pattern over 3+ consecutive readings
✓ Logical cause-effect relationship (e.g., high current → high temperature)
✓ Pattern matches known failure modes (bearing wear, seal failure, etc.)

SENSOR_GLITCH indicators:
✓ Single sudden spike that returns to normal immediately
✓ Physically impossible value (e.g., 60°C temperature jump in 1 second)
✓ Only one sensor showing anomaly while correlated sensors are normal
✓ Value outside theoretical physical limits
✓ Pattern inconsistent with equipment thermal/mechanical mass

NORMAL_WEAR indicators:
✓ Values slightly outside normal range but well below fault thresholds
✓ Gradual drift over extended period (days/weeks)
✓ No correlated anomalies in related sensors
✓ Pattern matches expected wear curves
✓ Equipment age/operating hours support gradual degradation

You MUST respond in JSON format with these exact keys:
{
    "ai_validation_status": "TRUE_FAULT" | "SENSOR_GLITCH" | "NORMAL_WEAR",
    "fault_category": "mechanical" | "thermal" | "electrical" | "process" | "sensor" | null,
    "confidence_score": 0.0-1.0,
    "ai_engineering_notes": "2-3 sentence technical explanation with root cause hypothesis and recommended action"
}

fault_category meanings:
- mechanical: Motors, pumps, bearings, vibration issues
- thermal: Temperature-related, overheating, cooling failure
- electrical: Current, voltage, power anomalies
- process: Flow, pressure, speed, throughput problems
- sensor: Sensor hardware malfunction (not transient glitch)
- null: For SENSOR_GLITCH or NORMAL_WEAR without specific fault

Be precise and actionable. Operators rely on your analysis to prioritize maintenance actions."""


FEW_SHOT_EXAMPLES = """
### EXAMPLE 1: TRUE_FAULT - Thermal Overload

INPUT:
Machine: MOTOR-101
ML Score: 0.85
Physics Violations:
  CRITICAL: thermistor_1: 38.2°C > fault_high (35°C)
  WARNING: lem_1: 57.1A > max_normal (55A)
Temporal Pattern:
  - Sustained over 5 readings
  - Trend: rising
  - No sudden spikes
Recent Readings:
  - thermistor_1: 38.2°C
  - lem_1: 57.1A
  - encoder_1: 1580 rpm (normal)
  - ct_1: 0.06A (normal)

OUTPUT:
{
    "ai_validation_status": "TRUE_FAULT",
    "fault_category": "thermal",
    "confidence_score": 0.92,
    "ai_engineering_notes": "Correlated thermal-electrical anomaly detected. Rising motor current (57.1A) causing overtemperature (38.2°C). Pattern indicates bearing degradation or cooling system failure. Immediate inspection recommended."
}

---

### EXAMPLE 2: SENSOR_GLITCH - EMI Spike

INPUT:
Machine: PUMP-002
ML Score: 0.78
Physics Violations:
  CRITICAL: thermistor_1: 85°C > fault_high (35°C) - IMPOSSIBLE JUMP from 25°C
Temporal Pattern:
  - Single spike (anomaly_count: 1)
  - Previous 4 readings normal
  - Trend: stable before spike
Recent Readings:
  - thermistor_1: 85°C (previous: 25°C)
  - lem_1: 50A (normal)
  - encoder_1: 1500 rpm (normal)
  - ct_1: 0.05A (normal)

OUTPUT:
{
    "ai_validation_status": "SENSOR_GLITCH",
    "fault_category": null,
    "confidence_score": 0.95,
    "ai_engineering_notes": "Temperature spike of 60°C in one reading is physically impossible given thermal mass. Only thermistor affected while all other sensors normal. Likely EMI interference or ADC glitch. No action required."
}

---

### EXAMPLE 3: NORMAL_WEAR - Gradual Speed Drift

INPUT:
Machine: CONVEYOR-003
ML Score: 0.45
Physics Violations:
  WARNING: encoder_1: 1620 rpm > max_normal (1600 rpm)
  (No critical violations)
Temporal Pattern:
  - Gradual increase over 10+ readings
  - Trend: stable (slow drift)
  - No spikes
Recent Readings:
  - encoder_1: 1620 rpm
  - thermistor_1: 26°C (normal)
  - lem_1: 48A (normal)
  - ct_1: 0.04A (normal)

OUTPUT:
{
    "ai_validation_status": "NORMAL_WEAR",
    "fault_category": null,
    "confidence_score": 0.88,
    "ai_engineering_notes": "Speed marginally elevated but well within fault threshold (1700 rpm). Gradual drift pattern consistent with belt wear or encoder calibration drift. All thermal and electrical parameters normal. Schedule maintenance check within 2 weeks."
}
"""


def build_validation_prompt(
    machine_id: str,
    ml_score: float,
    physics_summary: dict,
    temporal_pattern: dict,
    recent_readings: dict,
    hybrid_confidence: float
) -> str:
    """
    Build a structured prompt for the AI validation agent.
    
    Args:
        machine_id: The machine identifier
        ml_score: Autoencoder reconstruction error score
        physics_summary: Output from sensor_config_loader.get_violation_summary()
        temporal_pattern: Output from TemporalAnalyzer.analyze()
        recent_readings: Current sensor values
        hybrid_confidence: Pre-calculated hybrid confidence score
        
    Returns:
        Formatted prompt string for GPT-4o
    """
    # Build physics violations section
    physics_text = physics_summary.get("summary_text", "No violations detected.")
    
    # Build temporal pattern section
    temporal_text = f"""Pattern Analysis:
  - Type: {"Sudden Spike" if temporal_pattern.get("is_spike") else "Sustained Trend" if temporal_pattern.get("is_sustained") else "Normal Variation"}
  - Consecutive Anomalies: {temporal_pattern.get("anomaly_count", 0)}
  - Trend Direction: {temporal_pattern.get("trend", "stable")}
  - Max Rate of Change: {temporal_pattern.get("max_rate_of_change", 0):.2f} sigma"""
    
    if temporal_pattern.get("suspicious_sensors"):
        temporal_text += "\n  - Suspicious Sensors: " + ", ".join(
            f"{s['sensor_id']} ({s['pattern']})" 
            for s in temporal_pattern["suspicious_sensors"]
        )
    
    # Build readings section
    readings_lines = []
    for sensor_id, value in recent_readings.items():
        try:
            readings_lines.append(f"  - {sensor_id}: {float(value):.2f}")
        except (TypeError, ValueError):
            readings_lines.append(f"  - {sensor_id}: {value}")
    readings_text = "\n".join(readings_lines) if readings_lines else "  No readings available"
    
    prompt = f"""Analyze this anomaly and provide your validation decision:

MACHINE: {machine_id}
ML ANOMALY SCORE: {ml_score:.3f}
HYBRID CONFIDENCE: {hybrid_confidence:.3f}

PHYSICS VALIDATION:
{physics_text}

TEMPORAL ANALYSIS:
{temporal_text}

CURRENT SENSOR READINGS:
{readings_text}

Based on your expertise, classify this anomaly and explain your reasoning. Remember to consider:
1. Are the physics violations critical or just warnings?
2. Is this a sudden spike (likely glitch) or sustained trend (likely real fault)?
3. Are multiple correlated sensors affected?
4. Is the pattern physically plausible?

Respond with a JSON object containing ai_validation_status, fault_category, confidence_score, and ai_engineering_notes."""

    return prompt


def get_full_system_prompt() -> str:
    """Return the complete system prompt including few-shot examples."""
    return VALIDATION_ENGINEER_SYSTEM_PROMPT + "\n\n" + FEW_SHOT_EXAMPLES
