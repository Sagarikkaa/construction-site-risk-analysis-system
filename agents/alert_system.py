"""
Safety Alert System
====================
Generates, de-duplicates, and manages real-time safety alerts for PPE violations
and high-risk site events.
"""

import os
import sys
import uuid
from datetime import datetime
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import SafetyDBManager
from utils.config import ALERT_COOLDOWN_SECONDS, ALERT_STATUSES, ALERT_SEVERITIES


class AlertSystem:
    """
    Manages safety alert generation with cooldown de-duplication and persistence.
    """

    def __init__(self, db_manager: Optional[SafetyDBManager] = None, cooldown_seconds: float = ALERT_COOLDOWN_SECONDS):
        self.db = db_manager or SafetyDBManager()
        self.cooldown_seconds = cooldown_seconds
        self._alert_counter = 1000

    def process_worker_violations(self, worker_results: List[Dict]) -> List[Dict]:
        """
        Evaluate worker compliance results and generate alerts for non-compliant workers.

        Parameters
        ----------
        worker_results : list of worker result dicts

        Returns
        -------
        list of generated alert dicts
        """
        generated_alerts = []

        for w in worker_results:
            if not w.get("alert", False) or w.get("ppe_status") == "Compliant":
                continue

            worker_id = w.get("worker_id", "Worker-01")
            missing_ppe = w.get("missing_ppe", [])
            confidence = w.get("confidence", 0.90)

            # Determine violation description and risk level
            if "helmet" in missing_ppe and "vest" in missing_ppe:
                violation_type = "Worker detected without required PPE (Missing Helmet and Vest)"
                risk_level = "CRITICAL" if w.get("risk_level") in ["Critical", "High"] else "HIGH"
            elif "helmet" in missing_ppe:
                violation_type = "Worker detected without helmet"
                risk_level = "HIGH"
            elif "vest" in missing_ppe:
                violation_type = "Worker detected without safety vest"
                risk_level = "MEDIUM"
            else:
                violation_type = f"Worker detected without PPE ({', '.join(missing_ppe)})"
                risk_level = w.get("risk_level", "MEDIUM").upper()

            # Generate cooldown key for de-duplication
            cooldown_key = f"{worker_id}:{violation_type}"

            # Check cooldown
            if self.db.check_alert_cooldown(cooldown_key, self.cooldown_seconds):
                # Suppress duplicate alert within cooldown period
                continue

            alert_id = f"ALT-{uuid.uuid4().hex[:6].upper()}"
            now_iso = datetime.now().isoformat()

            alert = {
                "alert_id": alert_id,
                "timestamp": now_iso,
                "violation_type": violation_type,
                "risk_level": risk_level,
                "worker_id": worker_id,
                "confidence": confidence,
                "status": "New",
                "cooldown_key": cooldown_key,
            }

            # Save to SQLite database
            self.db.save_alert(alert)
            generated_alerts.append(alert)

        return generated_alerts

    def get_all_alerts(self, status: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Fetch alerts from the database."""
        return self.db.get_alerts(status=status, limit=limit)

    def update_alert_status(self, alert_id: str, new_status: str) -> bool:
        """Update an alert's status ('New', 'Acknowledged', 'Resolved')."""
        if new_status not in ALERT_STATUSES:
            raise ValueError(f"Invalid status '{new_status}'. Allowed: {ALERT_STATUSES}")
        return self.db.update_alert_status(alert_id, new_status)
