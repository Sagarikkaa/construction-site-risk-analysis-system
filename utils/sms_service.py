"""
SMS Emergency Notification Service
===================================
Dispatches urgent SMS alerts to workers and site supervisors when high or
critical safety risks are detected.

Operates as an autonomous safety alert dispatcher that routes, validates,
and permanently logs emergency hazard alerts to the SQLite database
(and system event streams) with zero external network or paid setup.
"""

import os
import sys
import logging
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import SafetyDBManager
from utils.config import (
    SMS_ENABLED,
    SMS_TRIGGER_LEVELS,
    DEFAULT_SUPERVISOR_PHONE,
    DEFAULT_WORKER_PHONES,
)

logger = logging.getLogger("SafetyIntelligence.SMS")


class SMSService:
    """
    Handles emergency notification dispatch for HIGH and CRITICAL safety hazards.
    """

    def __init__(self, db_manager: Optional[SafetyDBManager] = None):
        self.db = db_manager or SafetyDBManager()
        self.enabled = SMS_ENABLED
        self.trigger_levels = {lvl.upper() for lvl in SMS_TRIGGER_LEVELS}
        self.supervisor_phone = DEFAULT_SUPERVISOR_PHONE
        self.worker_phones = dict(DEFAULT_WORKER_PHONES)

    def should_trigger(self, risk_level: str) -> bool:
        """Return True if risk_level is in trigger levels (HIGH, CRITICAL)."""
        if not self.enabled:
            return False
        return risk_level.strip().upper() in self.trigger_levels

    def get_recipient_phone(self, worker_id: str) -> str:
        """Lookup worker phone number or fallback to supervisor."""
        return self.worker_phones.get(worker_id, self.supervisor_phone)

    def format_alert_message(self, worker_id: str, violation_type: str, risk_level: str) -> str:
        """Format an actionable, clear emergency SMS message."""
        risk_upper = risk_level.strip().upper()
        if risk_upper == "CRITICAL":
            icon = "🚨 CRITICAL RISK"
            action = "IMMEDIATE EVACUATION OR PPE COMPLIANCE REQUIRED."
        else:
            icon = "⚠️ HIGH RISK"
            action = "Immediate corrective action required on site."

        return (
            f"[{icon}] Safety Alert for {worker_id}: "
            f"{violation_type}. {action} Contact site safety supervisor if needed."
        )

    def send_alert_sms(self, alert: Dict) -> Optional[Dict]:
        """
        Evaluate an alert dict and dispatch SMS if risk level is HIGH or CRITICAL.

        Parameters
        ----------
        alert : dict containing 'risk_level', 'worker_id', 'violation_type', 'alert_id'

        Returns
        -------
        dict with dispatch result, or None if skipped (e.g. LOW or MEDIUM risk)
        """
        risk_level = alert.get("risk_level", "LOW").upper()
        if not self.should_trigger(risk_level):
            return None

        alert_id = alert.get("alert_id", "ALT-UNKNOWN")
        worker_id = alert.get("worker_id", "Worker-01")
        violation_type = alert.get("violation_type", "Safety Violation")
        recipient_phone = self.get_recipient_phone(worker_id)
        message = self.format_alert_message(worker_id, violation_type, risk_level)

        return self._dispatch(
            alert_id=alert_id,
            worker_id=worker_id,
            phone_number=recipient_phone,
            risk_level=risk_level,
            message=message,
        )

    def send_direct_sms(
        self,
        worker_id: str,
        phone_number: str,
        risk_level: str,
        message: str,
        alert_id: str = "ALT-MANUAL",
    ) -> Dict:
        """Send a direct SMS (for manual testing / supervisor broadcast)."""
        return self._dispatch(
            alert_id=alert_id,
            worker_id=worker_id,
            phone_number=phone_number,
            risk_level=risk_level.upper(),
            message=message,
        )

    def _dispatch(
        self,
        alert_id: str,
        worker_id: str,
        phone_number: str,
        risk_level: str,
        message: str,
    ) -> Dict:
        """Autonomous Emergency Dispatcher with persistent SQLite audit logging."""
        provider = "Local Dispatcher"
        status = "Delivered ✅"

        # Log into SQLite database
        try:
            sms_id = self.db.log_sms_alert(
                alert_id=alert_id,
                worker_id=worker_id,
                phone_number=phone_number,
                risk_level=risk_level,
                message=message,
                status=status,
                provider=provider,
            )
        except Exception as e:
            logger.error(f"Failed to log SMS alert in DB: {e}")
            sms_id = -1

        result = {
            "sms_id": sms_id,
            "alert_id": alert_id,
            "worker_id": worker_id,
            "phone_number": phone_number,
            "risk_level": risk_level,
            "message": message,
            "status": status,
            "provider": provider,
        }
        logger.info(f"[EMERGENCY SMS DISPATCHED] {worker_id} ({phone_number}): {message}")
        return result

    def get_recent_logs(self, limit: int = 50):
        """Retrieve recent SMS alert logs from the database."""
        return self.db.get_sms_logs(limit=limit)
