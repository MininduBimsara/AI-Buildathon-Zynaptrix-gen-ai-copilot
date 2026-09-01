"""
anomaly_service.py — Real-time anomaly detection pipeline service.

Pipeline:
    new sensor reading (dict)
        ↓  normalize
        ↓  AnomalyDetector.detect()
        ↓  TemporalAnalyzer.analyze()
        ↓  calculate_hybrid_confidence()
        ↓  if anomaly → alert_service.trigger()
        ↓  log result

Used by:
  - The real-time simulator (sensor_simulator.py)
  - The FastAPI anomaly endpoint (api/anomaly_routes.py)
"""

import logging
from datetime import datetime, timezone
from typing import Callable, Dict, Any, List, Optional
from collections import defaultdict
import statistics

from models.detect_anomaly import AnomalyDetector
from services.alert_service import format_alert, log_alert
from services.sensor_config_loader import sensor_config_loader

log = logging.getLogger(__name__)


class TemporalAnalyzer:
    """
    Analyzes sensor reading patterns over time to distinguish spikes from sustained trends.
    Tracks the last N readings per machine/sensor pair.
    """
    
    def __init__(self, history_size: int = 5):
        """
        Args:
            history_size: Number of recent readings to track per sensor
        """
        self._history_size = history_size
        # Structure: {machine_id: {sensor_id: [readings]}}
        self._history: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        # Track anomaly counts per machine
        self._anomaly_counts: Dict[str, int] = defaultdict(int)
    
    def add_reading(self, machine_id: str, sensor_id: str, value: float) -> None:
        """Add a new reading to the history buffer."""
        history = self._history[machine_id][sensor_id]
        history.append(value)
        if len(history) > self._history_size:
            history.pop(0)
    
    def add_readings(self, machine_id: str, readings: Dict[str, float]) -> None:
        """Add multiple sensor readings at once."""
        for sensor_id, value in readings.items():
            try:
                self.add_reading(machine_id, sensor_id, float(value))
            except (TypeError, ValueError):
                pass
    
    def increment_anomaly_count(self, machine_id: str) -> None:
        """Increment the anomaly counter for a machine."""
        self._anomaly_counts[machine_id] += 1
    
    def reset_anomaly_count(self, machine_id: str) -> None:
        """Reset the anomaly counter (called when reading is normal)."""
        self._anomaly_counts[machine_id] = 0
    
    def analyze(self, machine_id: str, readings: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze temporal patterns for the given readings.
        
        Returns:
            dict with keys:
                - is_spike: bool - Single sudden deviation
                - is_sustained: bool - Consistent trend over multiple readings
                - anomaly_count: int - Consecutive anomalous readings
                - trend: 'rising' | 'falling' | 'erratic' | 'stable'
                - max_rate_of_change: float - Maximum change rate across sensors
                - suspicious_sensors: list - Sensors with unusual patterns
        """
        suspicious_sensors = []
        max_rate_of_change = 0.0
        trends = []
        
        for sensor_id, current_value in readings.items():
            try:
                current = float(current_value)
            except (TypeError, ValueError):
                continue
            
            history = self._history[machine_id].get(sensor_id, [])
            
            if len(history) < 2:
                continue
            
            # Calculate statistics
            avg = statistics.mean(history)
            stdev = statistics.stdev(history) if len(history) > 1 else 0.1
            if stdev == 0:
                stdev = 0.1
            
            # Deviation from moving average
            deviation = abs(current - avg) / stdev
            
            # Rate of change from last reading
            rate_of_change = abs(current - history[-1]) / stdev if history else 0
            max_rate_of_change = max(max_rate_of_change, rate_of_change)
            
            # Check for sudden spike (current >> historical average)
            if deviation > 3.0 and rate_of_change > 2.0:
                suspicious_sensors.append({
                    "sensor_id": sensor_id,
                    "pattern": "sudden_spike",
                    "deviation": round(deviation, 2)
                })
            
            # Determine trend direction
            if len(history) >= 3:
                diffs = [history[i] - history[i-1] for i in range(1, len(history))]
                diffs.append(current - history[-1])
                
                positive_count = sum(1 for d in diffs if d > 0)
                negative_count = sum(1 for d in diffs if d < 0)
                
                if positive_count >= len(diffs) * 0.8:
                    trends.append("rising")
                elif negative_count >= len(diffs) * 0.8:
                    trends.append("falling")
                else:
                    trends.append("erratic" if max(abs(d) for d in diffs) > stdev else "stable")
        
        # Determine overall trend
        if trends:
            trend_counts = {"rising": 0, "falling": 0, "erratic": 0, "stable": 0}
            for t in trends:
                trend_counts[t] += 1
            overall_trend = max(trend_counts, key=trend_counts.get)
        else:
            overall_trend = "stable"
        
        anomaly_count = self._anomaly_counts.get(machine_id, 0)
        
        # Determine if spike or sustained
        is_spike = len(suspicious_sensors) > 0 and anomaly_count <= 1
        is_sustained = anomaly_count >= 3
        
        return {
            "is_spike": is_spike,
            "is_sustained": is_sustained,
            "anomaly_count": anomaly_count,
            "trend": overall_trend,
            "max_rate_of_change": round(max_rate_of_change, 2),
            "suspicious_sensors": suspicious_sensors
        }


def calculate_hybrid_confidence(
    ml_score: float,
    physics_summary: Dict[str, Any],
    temporal_pattern: Dict[str, Any]
) -> float:
    """
    Combines ML score, physics violations, and temporal patterns into unified confidence.
    
    Formula:
        hybrid_confidence = ML_score 
                          + (0.3 if physics_violation) 
                          + (0.2 if sustained_trend)
                          - (0.15 if sudden_spike)
    
    Args:
        ml_score: MSE score from autoencoder (0-1)
        physics_summary: Output from sensor_config_loader.get_violation_summary()
        temporal_pattern: Output from TemporalAnalyzer.analyze()
    
    Returns:
        Normalized confidence score [0-1]
    """
    confidence = ml_score
    
    # Boost confidence if physics limits violated
    if physics_summary.get("fault_violations"):
        confidence += 0.3  # Critical violations
    elif physics_summary.get("normal_violations"):
        confidence += 0.15  # Warning violations
    
    # Boost if sustained trend detected
    if temporal_pattern.get("is_sustained"):
        confidence += 0.2
    
    # Reduce if sudden spike (likely sensor glitch)
    if temporal_pattern.get("is_spike"):
        confidence -= 0.15
    
    # Normalize to [0, 1]
    return max(0.0, min(1.0, confidence))


class AnomalyService:
    """
    Stateful service wrapping AnomalyDetector.
    Maintains a rolling count of recent anomalies and exposes
    a callback hook so callers can react to events.
    """

    def __init__(self,
                 on_anomaly: Callable[[dict], None] | None = None,
                 consecutive_threshold: int = 3):
        """
        Args:
            on_anomaly:              Optional callback called with alert dict on anomaly.
            consecutive_threshold:   Number of consecutive anomaly readings before
                                     escalating (future use by agent layer).
        """
        self._detector   = AnomalyDetector()
        self._on_anomaly = on_anomaly
        self._consec_threshold = consecutive_threshold
        self._consecutive_count = 0
        self._total_processed   = 0
        self._total_anomalies   = 0

    def process(self, reading: dict) -> dict:
        """
        Process one sensor reading through the anomaly detection pipeline.

        Args:
            reading: dict with sensor keys matching SENSOR_COLUMNS.

        Returns:
            result dict from AnomalyDetector.detect() augmented with:
              - timestamp
              - consecutive_anomalies
              - escalated (bool)
        """
        result = self._detector.detect(reading)
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        health_score = result.get("health_score", 100)

        self._total_processed += 1

        if result["is_anomaly"]:
            self._consecutive_count += 1
            self._total_anomalies   += 1
            result["consecutive_anomalies"] = self._consecutive_count
            result["escalated"] = self._consecutive_count >= self._consec_threshold

            alert = format_alert(result)
            alert["health_score"] = health_score
            log_alert(alert)

            if self._on_anomaly:
                self._on_anomaly(alert)
        else:
            self._consecutive_count = 0
            result["consecutive_anomalies"] = 0
            result["escalated"] = False

        log.debug(
            f"[{'ANOMALY' if result['is_anomaly'] else 'OK':7s}] "
            f"score={result['score']:.5f}  "
            f"threshold={result['threshold']:.5f}"
        )
        return result

    @property
    def stats(self) -> dict:
        return {
            "total_processed":   self._total_processed,
            "total_anomalies":   self._total_anomalies,
            "anomaly_rate_pct":  round(
                self._total_anomalies / max(self._total_processed, 1) * 100, 2
            ),
        }
