"""
Enterprise Escalation Service
===============================
Routes high-severity incidents into automated escalation workflows:
  - Email (SMTP)
  - SMS (via existing SMSService)
  - Slack / Microsoft Teams webhooks

All dispatch attempts are audit-logged regardless of delivery success.
External services degrade gracefully — failures are recorded but never
block the calling pipeline.
"""

import json
import os
import smtplib
import sys
import uuid
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    ESCALATION_RULES,
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD,
    SMTP_FROM_EMAIL, ESCALATION_EMAIL_RECIPIENTS,
    SLACK_WEBHOOK_URL, TEAMS_WEBHOOK_URL,
    SMS_TRIGGER_LEVELS,
)


class EscalationService:
    """
    Severity-based escalation dispatcher.

    Routing rules (configurable in config.py):
        LOW      → log only
        MEDIUM   → log + email
        HIGH     → log + email + SMS
        CRITICAL → log + email + SMS + Slack/Teams webhooks
    """

    def __init__(self, db_manager=None, sms_service=None):
        self.db = db_manager
        self.sms_service = sms_service
        self._rules = ESCALATION_RULES

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def escalate(self, alert: Dict) -> Dict:
        """
        Execute the full escalation workflow for an alert.

        Parameters
        ----------
        alert : dict
            Must contain at minimum: alert_id, severity (or risk_level),
            alert_type (or violation_type), project_id.

        Returns
        -------
        dict — escalation result with channels attempted and their status.
        """
        severity = (
            alert.get("severity") or alert.get("risk_level") or "LOW"
        ).upper()
        channels = self._rules.get(severity, ["log"])
        results: Dict[str, str] = {}
        timestamp = datetime.now().isoformat()

        # Always log
        self._log_audit(
            action="ESCALATION_INITIATED",
            details={
                "alert_id": alert.get("alert_id", "UNKNOWN"),
                "severity": severity,
                "channels": channels,
                "timestamp": timestamp,
            },
        )
        results["log"] = "recorded"

        # Email
        if "email" in channels:
            try:
                self._send_email(alert, severity)
                results["email"] = "sent"
            except Exception as exc:
                results["email"] = f"failed: {exc}"

        # SMS
        if "sms" in channels:
            try:
                self._send_sms(alert)
                results["sms"] = "sent"
            except Exception as exc:
                results["sms"] = f"failed: {exc}"

        # Webhook (Slack + Teams)
        if "webhook" in channels:
            for name, url in [("slack", SLACK_WEBHOOK_URL), ("teams", TEAMS_WEBHOOK_URL)]:
                if url:
                    try:
                        self._send_webhook(url, alert, severity, platform=name)
                        results[name] = "sent"
                    except Exception as exc:
                        results[name] = f"failed: {exc}"
                else:
                    results[name] = "skipped (no URL configured)"

        # Final audit log
        self._log_audit(
            action="ESCALATION_COMPLETED",
            details={
                "alert_id": alert.get("alert_id", "UNKNOWN"),
                "severity": severity,
                "results": results,
                "timestamp": datetime.now().isoformat(),
            },
        )

        return {
            "alert_id": alert.get("alert_id", "UNKNOWN"),
            "severity": severity,
            "channels_attempted": channels,
            "results": results,
            "escalated_at": timestamp,
        }

    # ------------------------------------------------------------------
    # Email Dispatch
    # ------------------------------------------------------------------
    def _send_email(self, alert: Dict, severity: str) -> None:
        """Send an HTML-formatted alert email via SMTP."""
        if not ESCALATION_EMAIL_RECIPIENTS:
            self._log_audit(
                action="EMAIL_SKIPPED",
                details={"reason": "No recipients configured"},
            )
            return

        subject = (
            f"[{severity}] Construction Risk Alert — "
            f"{alert.get('alert_type', alert.get('violation_type', 'Safety Alert'))}"
        )

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <div style="background: {'#dc2626' if severity == 'CRITICAL' else '#f97316' if severity == 'HIGH' else '#eab308'};
                        color: white; padding: 15px; border-radius: 8px;">
                <h2>⚠️ {severity} Safety Alert</h2>
            </div>
            <table style="margin-top: 15px; border-collapse: collapse; width: 100%;">
                <tr><td style="padding: 8px; font-weight: bold;">Alert ID:</td>
                    <td style="padding: 8px;">{alert.get('alert_id', 'N/A')}</td></tr>
                <tr><td style="padding: 8px; font-weight: bold;">Project:</td>
                    <td style="padding: 8px;">{alert.get('project_id', 'N/A')}</td></tr>
                <tr><td style="padding: 8px; font-weight: bold;">Type:</td>
                    <td style="padding: 8px;">{alert.get('alert_type', alert.get('violation_type', 'N/A'))}</td></tr>
                <tr><td style="padding: 8px; font-weight: bold;">Severity:</td>
                    <td style="padding: 8px;">{severity}</td></tr>
                <tr><td style="padding: 8px; font-weight: bold;">Timestamp:</td>
                    <td style="padding: 8px;">{alert.get('created_at', datetime.now().isoformat())}</td></tr>
            </table>
            <p style="margin-top: 20px; color: #6b7280; font-size: 12px;">
                Construction Risk Intelligence Platform — Automated Escalation
            </p>
        </body>
        </html>
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = SMTP_FROM_EMAIL
        msg["To"] = ", ".join(ESCALATION_EMAIL_RECIPIENTS)
        msg.attach(MIMEText(html_body, "html"))

        if not SMTP_HOST or SMTP_HOST == "localhost":
            # Local-dev mode: log instead of sending
            self._log_audit(
                action="EMAIL_SIMULATED",
                details={
                    "subject": subject,
                    "recipients": ESCALATION_EMAIL_RECIPIENTS,
                    "alert_id": alert.get("alert_id"),
                },
            )
            return

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            if SMTP_USER and SMTP_PASSWORD:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)

    # ------------------------------------------------------------------
    # SMS Dispatch
    # ------------------------------------------------------------------
    def _send_sms(self, alert: Dict) -> None:
        """Dispatch SMS using the existing SMSService."""
        if self.sms_service:
            sms_alert = {
                "alert_id": alert.get("alert_id", f"ESC-{uuid.uuid4().hex[:6].upper()}"),
                "timestamp": alert.get("created_at", datetime.now().isoformat()),
                "violation_type": alert.get("alert_type", alert.get("violation_type", "Safety Alert")),
                "risk_level": (alert.get("severity") or alert.get("risk_level", "HIGH")).upper(),
                "worker_id": alert.get("worker_id", f"Project-{alert.get('project_id', 'UNKNOWN')}"),
                "confidence": float(alert.get("confidence", 1.0)),
                "status": "New",
                "cooldown_key": f"ESC:{alert.get('alert_id', 'UNKNOWN')}",
            }
            self.sms_service.send_alert_sms(sms_alert)

    # ------------------------------------------------------------------
    # Webhook Dispatch (Slack / Teams)
    # ------------------------------------------------------------------
    def _send_webhook(self, url: str, alert: Dict, severity: str, platform: str = "slack") -> None:
        """POST a structured payload to a Slack or Teams webhook."""
        if platform == "teams":
            payload = self._teams_payload(alert, severity)
        else:
            payload = self._slack_payload(alert, severity)

        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            urlopen(req, timeout=10)
        except URLError as exc:
            self._log_audit(
                action="WEBHOOK_FAILED",
                details={"platform": platform, "error": str(exc), "alert_id": alert.get("alert_id")},
            )
            raise

    @staticmethod
    def _slack_payload(alert: Dict, severity: str) -> Dict:
        """Build a Slack Block Kit message payload."""
        color = {"CRITICAL": "#dc2626", "HIGH": "#f97316", "MEDIUM": "#eab308"}.get(severity, "#6b7280")
        return {
            "attachments": [{
                "color": color,
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": f"⚠️ {severity} Construction Risk Alert"},
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Alert ID:*\n{alert.get('alert_id', 'N/A')}"},
                            {"type": "mrkdwn", "text": f"*Project:*\n{alert.get('project_id', 'N/A')}"},
                            {"type": "mrkdwn", "text": f"*Type:*\n{alert.get('alert_type', alert.get('violation_type', 'N/A'))}"},
                            {"type": "mrkdwn", "text": f"*Severity:*\n{severity}"},
                        ],
                    },
                ],
            }],
        }

    @staticmethod
    def _teams_payload(alert: Dict, severity: str) -> Dict:
        """Build a Microsoft Teams Adaptive Card payload."""
        color = {"CRITICAL": "attention", "HIGH": "warning", "MEDIUM": "accent"}.get(severity, "default")
        return {
            "@type": "MessageCard",
            "themeColor": {"CRITICAL": "DC2626", "HIGH": "F97316", "MEDIUM": "EAB308"}.get(severity, "6B7280"),
            "summary": f"{severity} Construction Risk Alert",
            "sections": [{
                "activityTitle": f"⚠️ {severity} Construction Risk Alert",
                "facts": [
                    {"name": "Alert ID", "value": alert.get("alert_id", "N/A")},
                    {"name": "Project", "value": alert.get("project_id", "N/A")},
                    {"name": "Type", "value": alert.get("alert_type", alert.get("violation_type", "N/A"))},
                    {"name": "Severity", "value": severity},
                ],
            }],
        }

    # ------------------------------------------------------------------
    # Audit Logging
    # ------------------------------------------------------------------
    def _log_audit(self, action: str, details: Dict) -> None:
        """Persist an audit trail entry if a database manager is available."""
        if self.db and hasattr(self.db, "log_audit_trail"):
            try:
                self.db.log_audit_trail(
                    action=action,
                    agent="EscalationService",
                    details=details,
                )
            except Exception:
                pass  # Never block the caller on audit-log failures
