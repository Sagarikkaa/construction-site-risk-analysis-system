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
from utils.sms_service import SMSService


class AlertSystem:
    """
    Manages safety alert generation with cooldown de-duplication, persistence,
    and automatic SMS dispatch for High and Critical hazards.
    """

    def __init__(self, db_manager: Optional[SafetyDBManager] = None, cooldown_seconds: float = ALERT_COOLDOWN_SECONDS):
        self.db = db_manager or SafetyDBManager()
        self.cooldown_seconds = cooldown_seconds
        self.sms_service = SMSService(db_manager=self.db)
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

            # Automated SMS Emergency Dispatch for HIGH and CRITICAL risks
            sms_result = self.sms_service.send_alert_sms(alert)
            if sms_result:
                alert["sms_sent"] = True
                alert["sms_recipient"] = sms_result.get("phone_number")
                alert["sms_provider"] = sms_result.get("provider")
                alert["sms_status"] = sms_result.get("status")
            else:
                alert["sms_sent"] = False

            generated_alerts.append(alert)

        return generated_alerts

    def process_site_risk(self, risk_report: Dict, source: str = "Site Area") -> Optional[Dict]:
        """
        Evaluate overall site risk report and dispatch critical site alert + SMS
        if overall site risk reaches HIGH or CRITICAL severity.
        """
        level = risk_report.get("level", "Low").upper()
        if level not in ["HIGH", "CRITICAL"]:
            return None

        cooldown_key = f"SITE:{level}:{source}"
        if self.db.check_alert_cooldown(cooldown_key, self.cooldown_seconds):
            return None

        alert_id = f"ALT-{uuid.uuid4().hex[:6].upper()}"
        now_iso = datetime.now().isoformat()
        violation_type = f"High Site Risk Event: Score {risk_report.get('score', 0)} ({risk_report.get('level', 'High')})"

        alert = {
            "alert_id": alert_id,
            "timestamp": now_iso,
            "violation_type": violation_type,
            "risk_level": level,
            "worker_id": "Site-Wide",
            "confidence": 0.95,
            "status": "New",
            "cooldown_key": cooldown_key,
        }
        self.db.save_alert(alert)

        sms_result = self.sms_service.send_alert_sms(alert)
        if sms_result:
            alert["sms_sent"] = True
            alert["sms_recipient"] = sms_result.get("phone_number")
            alert["sms_provider"] = sms_result.get("provider")
            alert["sms_status"] = sms_result.get("status")
        else:
            alert["sms_sent"] = False

        return alert

    def get_all_alerts(self, status: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Fetch alerts from the database."""
        return self.db.get_alerts(status=status, limit=limit)

    def update_alert_status(self, alert_id: str, new_status: str) -> bool:
        """Update an alert's status ('New', 'Acknowledged', 'Resolved')."""
        if new_status not in ALERT_STATUSES:
            raise ValueError(f"Invalid status '{new_status}'. Allowed: {ALERT_STATUSES}")
        return self.db.update_alert_status(alert_id, new_status)
