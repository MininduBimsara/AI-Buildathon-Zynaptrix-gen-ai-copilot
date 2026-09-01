"""
ai_automation_engineer.py — AI Automation Engineer Agent for Industrial Copilot.

This agent provides expert-level validation and generation capabilities:
1. Validates sensor configurations extracted from datasheets
2. Generates realistic anomaly patterns for training data
3. Cross-validates sensor relationships for physical plausibility
4. Provides high-accuracy fault classification

The agent acts as a Senior Industrial Automation Engineer with 20+ years of experience.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from unified_rag.ai_client import get_client, MODEL_CHAT
from services.llm_json import loads_tolerant

logger = logging.getLogger(__name__)


class AIAutomationEngineerAgent:
    """
    AI-powered Industrial Automation Engineer for high-accuracy validation
    and intelligent data generation.
    """

    def __init__(self):
        """Initialize the AI Automation Engineer Agent."""
        self.client = get_client()
        self._model = MODEL_CHAT
        self._temperature = 0.1  # Low for consistent engineering decisions
        
    def validate_sensor_config(
        self, 
        sensor_config: Dict[str, Any],
        datasheet_text: str = "",
        machine_type: str = ""
    ) -> Dict[str, Any]:
        """
        Validate extracted sensor configuration using AI engineering expertise.
        
        Args:
            sensor_config: The extracted sensor configuration dict
            datasheet_text: Original datasheet text for cross-reference
            machine_type: Type of machine (e.g., "industrial_motor", "pump", "conveyor")
            
        Returns:
            dict with:
                - is_valid: bool
                - confidence: float (0-1)
                - issues: list of identified issues
                - corrections: dict of suggested corrections
                - engineering_notes: str explanation
        """
        if not self.client:
            logger.warning("OpenAI client not available - skipping AI validation")
            return self._default_validation_result(sensor_config)
        
        prompt = f"""You are a Senior Industrial Automation Engineer validating sensor configuration extracted from a datasheet.

SENSOR CONFIGURATION TO VALIDATE:
{json.dumps(sensor_config, indent=2)}

MACHINE TYPE: {machine_type or "Industrial Equipment"}

ORIGINAL DATASHEET TEXT (first 3000 chars):
{datasheet_text[:3000] if datasheet_text else "[No datasheet provided]"}

VALIDATION CHECKLIST:
1. Physical Plausibility:
   - Is mu (mean) realistic for this sensor type?
   - Is sigma (std dev) reasonable (typically 1-10% of mu)?
   - Are min_normal/max_normal within manufacturer specs?
   
2. Fault Threshold Logic:
   - Is fault_high sufficiently above max_normal (typically 20-50% above)?
   - Is fault_low (if applicable) below min_normal?
   - Does fault_direction match the sensor physics?

3. Unit Consistency:
   - Is the unit correct for this sensor type?
   - Are all values in the same unit?

4. Cross-Sensor Relationships:
   - For temperature sensors: mu should be in realistic operating range (20-200°C typical)
   - For current sensors: mu depends on motor size
   - For vibration: mu typically 0.1-5 mm/s for healthy equipment
   - For pressure: depends on system (bar, psi, kPa)
   - For speed: RPM depends on equipment type

5. Icon Type Validation:
   - Does icon_type match the sensor measurement type?

Return JSON:
{{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "issues": ["list of identified problems"],
    "corrections": {{
        "field_name": "corrected_value"
    }},
    "engineering_notes": "2-3 sentence technical explanation",
    "validated_config": {{...corrected config if changes needed...}}
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a Senior Industrial Automation Engineer with expertise in sensor systems, instrumentation, and predictive maintenance."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self._temperature,
                max_tokens=3000,
                response_format={"type": "json_object"}
            )
            
            result = loads_tolerant(response.choices[0].message.content)
            if not isinstance(result, dict):
                raise ValueError("LLM returned unparseable JSON")
            logger.info(f"✅ [AI Engineer] Validated sensor config: valid={result.get('is_valid')}, confidence={result.get('confidence', 0):.2f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ [AI Engineer] Validation failed: {e}")
            return self._default_validation_result(sensor_config)
    
    def generate_anomaly_patterns(
        self,
        sensor_configs: Dict[str, Dict[str, Any]],
        machine_type: str = "",
        anomaly_type: str = "machine_fault"
    ) -> Dict[str, Any]:
        """
        Generate realistic anomaly patterns using AI engineering knowledge.
        
        Args:
            sensor_configs: Dict of sensor_id -> sensor_config
            machine_type: Type of machine
            anomaly_type: Type of anomaly to generate (machine_fault, sensor_drift, etc.)
            
        Returns:
            dict with:
                - patterns: list of generated reading patterns
                - correlations: expected sensor correlations during fault
                - progression: how fault develops over time
                - engineering_rationale: explanation of the physics
        """
        if not self.client:
            return self._default_anomaly_patterns(sensor_configs, anomaly_type)
        
        prompt = f"""You are a Senior Industrial Automation Engineer generating realistic anomaly patterns for training an ML model.

SENSOR CONFIGURATIONS:
{json.dumps(sensor_configs, indent=2)}

MACHINE TYPE: {machine_type or "Industrial Equipment"}
ANOMALY TYPE: {anomaly_type}

Generate realistic fault patterns based on your engineering knowledge:

For "{anomaly_type}":
1. Which sensors should show anomalies first? (primary indicators)
2. Which sensors will follow? (secondary effects)
3. What correlations exist? (e.g., current↑ → temperature↑)
4. How does the fault progress over time?
5. What are realistic magnitude changes?

Return JSON:
{{
    "patterns": [
        {{
            "stage": 1,
            "description": "Initial fault indication",
            "sensor_changes": {{
                "sensor_id": {{
                    "direction": "increase/decrease",
                    "magnitude_factor": 1.2,
                    "noise_factor": 1.5
                }}
            }},
            "duration_readings": 10
        }}
    ],
    "correlations": [
        {{
            "primary": "sensor_id_1",
            "secondary": "sensor_id_2", 
            "relationship": "positive/negative/lagged",
            "lag_readings": 0
        }}
    ],
    "progression_curve": "linear/exponential/step",
    "engineering_rationale": "Technical explanation of why this fault manifests this way"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a Senior Industrial Automation Engineer with expertise in failure modes and effects analysis (FMEA)."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Slightly higher for creative pattern generation
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            
            result = loads_tolerant(response.choices[0].message.content)
            if not isinstance(result, dict):
                raise ValueError("LLM returned unparseable JSON")
            logger.info(f"✅ [AI Engineer] Generated {len(result.get('patterns', []))} anomaly patterns for {anomaly_type}")
            return result
            
        except Exception as e:
            logger.error(f"❌ [AI Engineer] Pattern generation failed: {e}")
            return self._default_anomaly_patterns(sensor_configs, anomaly_type)
    
    def cross_validate_sensors(
        self,
        sensor_configs: Dict[str, Dict[str, Any]],
        machine_type: str = ""
    ) -> Dict[str, Any]:
        """
        Cross-validate sensor configurations for physical consistency.
        
        Checks:
        - Sensor relationships make physical sense
        - No conflicting operating ranges
        - Expected correlations are defined
        """
        if not self.client:
            return {"is_consistent": True, "issues": [], "recommendations": []}
        
        prompt = f"""You are a Senior Industrial Automation Engineer reviewing sensor configurations for physical consistency.

SENSOR CONFIGURATIONS:
{json.dumps(sensor_configs, indent=2)}

MACHINE TYPE: {machine_type or "Industrial Equipment"}

Cross-validation checks:
1. Do temperature and current sensors have correlated operating ranges?
2. Are vibration limits appropriate for the speed range?
3. Do pressure and flow sensors (if present) have consistent relationships?
4. Are any sensors redundant or missing for proper fault detection?
5. Would these sensors provide adequate fault coverage?

Return JSON:
{{
    "is_consistent": true/false,
    "consistency_score": 0.0-1.0,
    "issues": ["list of inconsistencies found"],
    "recommendations": ["list of recommendations"],
    "expected_correlations": [
        {{"sensor_1": "id", "sensor_2": "id", "correlation": "positive/negative", "strength": 0.0-1.0}}
    ],
    "fault_coverage_score": 0.0-1.0,
    "engineering_summary": "Brief technical summary"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a Senior Industrial Automation Engineer specializing in sensor system design and fault detection."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self._temperature,
                max_tokens=3000,
                response_format={"type": "json_object"}
            )
            
            result = loads_tolerant(response.choices[0].message.content)
            if not isinstance(result, dict):
                raise ValueError("LLM returned unparseable JSON")
            logger.info(f"✅ [AI Engineer] Cross-validation complete: consistent={result.get('is_consistent')}")
            return result
            
        except Exception as e:
            logger.error(f"❌ [AI Engineer] Cross-validation failed: {e}")
            return {"is_consistent": True, "issues": [], "recommendations": []}
    
    def high_accuracy_fault_classification(
        self,
        anomaly_data: Dict[str, Any],
        sensor_configs: Dict[str, Dict[str, Any]],
        historical_context: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Perform high-accuracy fault classification with detailed analysis.
        
        This is an enhanced version of validation_engineer_node that provides:
        - Multi-hypothesis analysis
        - Confidence intervals
        - Recommended actions with priority
        - Root cause probability distribution
        """
        if not self.client:
            return self._default_classification(anomaly_data)
        
        prompt = f"""You are a Senior Industrial Automation Engineer performing high-accuracy fault classification.

ANOMALY DATA:
- ML Score: {anomaly_data.get('anomaly_score', 0)}
- Machine ID: {anomaly_data.get('machine_id', 'UNKNOWN')}
- Recent Readings: {json.dumps(anomaly_data.get('recent_readings', {}), indent=2)}
- Physics Summary: {json.dumps(anomaly_data.get('physics_summary', {}), indent=2)}
- Temporal Pattern: {json.dumps(anomaly_data.get('temporal_pattern', {}), indent=2)}

SENSOR CONFIGURATIONS:
{json.dumps(sensor_configs, indent=2)}

HISTORICAL CONTEXT:
{json.dumps(historical_context[-5:] if historical_context else [], indent=2)}

Perform comprehensive fault analysis:

1. PRIMARY CLASSIFICATION (most likely):
   - TRUE_FAULT / SENSOR_GLITCH / NORMAL_WEAR

2. HYPOTHESIS RANKING:
   - List top 3 possible causes with probability %

3. CONFIDENCE ANALYSIS:
   - Overall confidence (0-1)
   - Evidence strength for each hypothesis

4. RECOMMENDED ACTIONS:
   - Immediate actions (if critical)
   - Diagnostic steps
   - Maintenance recommendations

5. ROOT CAUSE ANALYSIS:
   - Most probable root cause
   - Contributing factors
   - Fault propagation path

Return JSON:
{{
    "primary_classification": "TRUE_FAULT/SENSOR_GLITCH/NORMAL_WEAR",
    "fault_category": "mechanical/thermal/electrical/process/sensor/null",
    "confidence_score": 0.0-1.0,
    "confidence_interval": [0.0, 0.0],
    "hypotheses": [
        {{
            "rank": 1,
            "description": "Bearing degradation",
            "probability": 0.75,
            "supporting_evidence": ["list of evidence"],
            "contradicting_evidence": ["list of contradictions"]
        }}
    ],
    "root_cause_analysis": {{
        "primary_cause": "description",
        "contributing_factors": ["list"],
        "fault_propagation": "cause → effect → effect"
    }},
    "recommended_actions": [
        {{
            "priority": "immediate/high/medium/low",
            "action": "description",
            "rationale": "why this action"
        }}
    ],
    "engineering_notes": "Detailed technical explanation (3-5 sentences)",
    "requires_human_review": true/false
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a Senior Industrial Automation Engineer with 20+ years of experience in predictive maintenance, root cause analysis, and failure mode identification."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self._temperature,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            
            result = loads_tolerant(response.choices[0].message.content)
            if not isinstance(result, dict):
                raise ValueError("LLM returned unparseable JSON")
            logger.info(f"✅ [AI Engineer] High-accuracy classification: {result.get('primary_classification')} (confidence: {result.get('confidence_score', 0):.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ [AI Engineer] Classification failed: {e}")
            return self._default_classification(anomaly_data)
    
    def _default_validation_result(self, sensor_config: Dict) -> Dict[str, Any]:
        """Fallback when AI is unavailable."""
        return {
            "is_valid": True,
            "confidence": 0.5,
            "issues": ["AI validation unavailable - using default acceptance"],
            "corrections": {},
            "engineering_notes": "Configuration accepted without AI validation. Manual review recommended.",
            "validated_config": sensor_config
        }
    
    def _default_anomaly_patterns(self, sensor_configs: Dict, anomaly_type: str) -> Dict[str, Any]:
        """Fallback anomaly patterns when AI is unavailable."""
        return {
            "patterns": [
                {
                    "stage": 1,
                    "description": f"Standard {anomaly_type} pattern",
                    "sensor_changes": {
                        sid: {"direction": "increase", "magnitude_factor": 1.3, "noise_factor": 1.5}
                        for sid in list(sensor_configs.keys())[:2]
                    },
                    "duration_readings": 20
                }
            ],
            "correlations": [],
            "progression_curve": "linear",
            "engineering_rationale": "Default pattern - AI generation unavailable"
        }
    
    def _default_classification(self, anomaly_data: Dict) -> Dict[str, Any]:
        """Fallback classification when AI is unavailable."""
        return {
            "primary_classification": "TRUE_FAULT",
            "fault_category": "mechanical",
            "confidence_score": anomaly_data.get('anomaly_score', 0.5),
            "confidence_interval": [0.3, 0.7],
            "hypotheses": [{"rank": 1, "description": "Unknown fault", "probability": 1.0}],
            "root_cause_analysis": {"primary_cause": "Undetermined", "contributing_factors": []},
            "recommended_actions": [{"priority": "high", "action": "Manual inspection required", "rationale": "AI unavailable"}],
            "engineering_notes": "AI classification unavailable. Manual review recommended.",
            "requires_human_review": True
        }


# Singleton instance
ai_engineer = AIAutomationEngineerAgent()
