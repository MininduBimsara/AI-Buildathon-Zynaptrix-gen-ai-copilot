"""
datasheet_parser.py — AI-powered sensor datasheet analyzer.

Extracts comprehensive sensor parameters from PDF datasheets using Qwen.
Output includes realistic operating ranges, fault thresholds, units, and
an icon classification for the dashboard UI.

Now integrated with AI Automation Engineer for validation.
"""
import fitz
import json
import logging
from typing import Dict, Any
from unified_rag.ai_client import get_client, MODEL_CHAT

logger = logging.getLogger(__name__)

# Canonical icon types the dashboard knows how to render
VALID_ICON_TYPES = {
    "temperature", "current", "vibration", "pressure", "speed",
    "flow", "voltage", "humidity", "distance", "load", "torque",
    "position", "power", "frequency", "light", "gas", "force",
    "conductivity", "ph", "weight", "angle", "counter", "generic"
}

class DatasheetParser:
    def __init__(self):
        self.client = get_client()

    def parse_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text content from raw PDF bytes."""
        text = ""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page in doc:
                text += page.get_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            return text

    def extract_sensor_config(self, sensor_name: str, sensor_id: str, datasheet_text: str) -> Dict[str, Any]:
        """
        Use OpenAI GPT-4o to extract a comprehensive sensor configuration from datasheet text.
        Now includes AI Automation Engineer validation.
        
        Returns a dict with:
        - sensor_id, sensor_name, unit
        - mu: normal operating mean
        - sigma: normal operating std deviation
        - min_normal: lower boundary of healthy range
        - max_normal: upper boundary of healthy range
        - fault_high: value that clearly indicates a HIGH-side fault
        - fault_low: value that clearly indicates a LOW-side fault (None if not applicable)
        - fault_direction: "high" | "low" | "both"
        - icon_type: one of the canonical icon types
        - ai_validation: validation results from AI Engineer
        """
        prompt = f"""You are a senior industrial instrumentation engineer.
Analyze the datasheet below for sensor: '{sensor_name}' (ID: '{sensor_id}').

Your task is to extract the most realistic industrial operating parameters.
Read ALL text carefully — look for: rated ranges, operating ranges, nominal values,
tolerances, alarms/trips, signal ranges, specifications tables.

Return EXACTLY this JSON (no extra text), with every field filled:
{{
  "sensor_id": "{sensor_id}",
  "sensor_name": "{sensor_name}",
  "unit": "<measurement unit, e.g. degC, A, mm/s, bar, RPM, V, %, L/min>",
  "mu": <float: typical/nominal normal operating value>,
  "sigma": <float: realistic standard deviation during normal operation (NOT the full range)>,
  "min_normal": <float: lowest acceptable normal value>,
  "max_normal": <float: highest acceptable normal value>,
  "fault_high": <float: value clearly representing a HIGH-side fault/alarm>,
  "fault_low": <float or null: value clearly representing a LOW-side fault (null if one-directional)>,
  "fault_direction": "<'high' | 'low' | 'both'>",
  "icon_type": "<one of exactly: temperature|current|vibration|pressure|speed|flow|voltage|humidity|distance|load|torque|position|power|frequency|light|gas|force|conductivity|ph|weight|angle|counter|generic>"
}}

Rules:
- mu should be the midpoint of the normal operating range, NOT the fault value
- sigma should be about 1/6 of the normal range width (so 3-sigma covers most of the range)
- fault_high should be 40-100% above max_normal (realistic alarm point)
- If no PDF text provided, estimate realistic values based purely on the sensor name and your engineering knowledge
- icon_type must EXACTLY match one of the listed options based on what the sensor measures

Datasheet text (first 5000 chars):
{datasheet_text[:5000] if datasheet_text.strip() else "[No datasheet provided — use sensor name only for estimation]"}
"""
        try:
            res = self.client.chat.completions.create(
                model=MODEL_CHAT,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.05  # Very low for consistent engineering values
            )
            data = json.loads(res.choices[0].message.content)

            # Sanitize and guarantee correct types
            mu = float(data.get("mu", 50.0))
            sigma = float(data.get("sigma", max(mu * 0.05, 1.0)))
            min_normal = float(data.get("min_normal", mu - 3 * sigma))
            max_normal = float(data.get("max_normal", mu + 3 * sigma))
            fault_high = float(data.get("fault_high", max_normal * 1.5))
            fault_low_raw = data.get("fault_low")
            fault_low = float(fault_low_raw) if fault_low_raw is not None else None
            icon_type = str(data.get("icon_type", "generic"))
            if icon_type not in VALID_ICON_TYPES:
                icon_type = "generic"

            config = {
                "sensor_id":       str(data.get("sensor_id", sensor_id)),
                "sensor_name":     str(data.get("sensor_name", sensor_name)),
                "unit":            str(data.get("unit", "units")),
                "mu":              mu,
                "sigma":           sigma,
                "min_normal":      min_normal,
                "max_normal":      max_normal,
                "fault_high":      fault_high,
                "fault_low":       fault_low,
                "fault_direction": str(data.get("fault_direction", "high")),
                "icon_type":       icon_type,
            }
            
            # AI Automation Engineer Validation
            try:
                from agents.ai_automation_engineer import AIAutomationEngineerAgent
                ai_engineer = AIAutomationEngineerAgent()
                validation = ai_engineer.validate_sensor_config(config, datasheet_text)
                
                if validation.get("is_valid") and validation.get("validated_config"):
                    # Apply AI corrections if any
                    validated = validation.get("validated_config", {})
                    for key in ["mu", "sigma", "min_normal", "max_normal", "fault_high", "fault_low"]:
                        if key in validated and validated[key] is not None:
                            config[key] = validated[key]
                
                config["ai_validation"] = {
                    "is_valid": validation.get("is_valid", True),
                    "confidence": validation.get("confidence", 0.5),
                    "issues": validation.get("issues", []),
                    "engineering_notes": validation.get("engineering_notes", "")
                }
                logger.info(f"✅ AI Validation for '{sensor_name}': valid={validation.get('is_valid')}, confidence={validation.get('confidence', 0):.2f}")
                
            except Exception as val_error:
                logger.warning(f"⚠️ AI validation skipped: {val_error}")
                config["ai_validation"] = {"is_valid": True, "confidence": 0.5, "issues": ["Validation skipped"]}
            
            logger.info(f"✅ Parsed config for '{sensor_name}': mu={mu}, sigma={sigma:.3f}, icon={icon_type}")
            return config

        except Exception as e:
            logger.error(f"Error extracting config via LLM: {e}")
            # Smart fallback: at least try to guess icon from name
            return self._fallback_config(sensor_id, sensor_name)

    def extract_sensor_config_no_pdf(self, sensor_name: str, sensor_id: str) -> Dict[str, Any]:
        """Extract config using ONLY the sensor name (no datasheet)."""
        return self.extract_sensor_config(sensor_name, sensor_id, "")

    def _fallback_config(self, sensor_id: str, sensor_name: str) -> Dict[str, Any]:
        """Minimal fallback when LLM fails — infers icon_type from name heuristics."""
        name_lower = sensor_name.lower()
        if any(k in name_lower for k in ["temp", "therm", "heat", "cold"]):
            icon_type = "temperature"
        elif any(k in name_lower for k in ["current", "amp", "ct"]):
            icon_type = "current"
        elif any(k in name_lower for k in ["vibrat", "accel", "shock"]):
            icon_type = "vibration"
        elif any(k in name_lower for k in ["pressure", "press", "psi", "bar"]):
            icon_type = "pressure"
        elif any(k in name_lower for k in ["speed", "rpm", "rotat", "encoder"]):
            icon_type = "speed"
        elif any(k in name_lower for k in ["flow", "flux"]):
            icon_type = "flow"
        elif any(k in name_lower for k in ["volt", "lem"]):
            icon_type = "voltage"
        elif any(k in name_lower for k in ["humidity", "humid", "moisture"]):
            icon_type = "humidity"
        elif any(k in name_lower for k in ["load", "weight", "force", "torque"]):
            icon_type = "load"
        else:
            icon_type = "generic"

        return {
            "sensor_id": sensor_id,
            "sensor_name": sensor_name,
            "unit": "units",
            "mu": 50.0,
            "sigma": 5.0,
            "min_normal": 35.0,
            "max_normal": 65.0,
            "fault_high": 90.0,
            "fault_low": None,
            "fault_direction": "high",
            "icon_type": icon_type,
            "ai_validation": {"is_valid": False, "confidence": 0.0, "issues": ["Fallback config used"]}
        }
