"""Central Milestone 3 orchestration for project risk."""

import uuid
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional

from agents.compliance_agent import ComplianceAgent
from agents.insurance_agent import InsuranceAgent
from database.db_manager import SafetyDBManager
from utils.sms_service import SMSService


class ConstructionRiskEngine:
    """Aggregate compliance and insurance signals and dispatch severe alerts."""

    def __init__(
        self,
        db_manager: Optional[SafetyDBManager] = None,
        notifier: Optional[Callable[[Dict], None]] = None,
    ):
        self.db = db_manager or SafetyDBManager()
        self.compliance_agent = ComplianceAgent(self.db)
        self.insurance_agent = InsuranceAgent(self.db)
        self.sms_service = SMSService(db_manager=self.db)
        self.notifier = notifier

    def assess_project(
        self,
        project_id: str,
        inspection_reports: Iterable[Dict] = (),
        safety_logs: Iterable[Dict] = (),
        incidents: Iterable[Dict] = (),
        safety_trends: Iterable[Dict] = (),
    ) -> Dict:
        """Run deterministic compliance and insurance assessments on submitted data."""
        if not project_id or not str(project_id).strip():
            raise ValueError("Project ID cannot be empty.")

        records = {
            "inspection_reports": list(inspection_reports),
            "safety_logs": list(safety_logs),
            "incidents": list(incidents),
            "safety_trends": list(safety_trends),
        }
        for name, values in records.items():
            if not all(isinstance(value, dict) for value in values):
                raise ValueError(f"{name} must be a JSON list of objects.")

        # 1. Compliance Agent Assessment
        compliance = self.compliance_agent.evaluate(
            project_id=project_id.strip(),
            inspection_reports=records["inspection_reports"],
            safety_logs=records["safety_logs"],
            safety_violation_trends=records["safety_trends"],
        )

        # 2. Insurance Agent Assessment
        insurance = self.insurance_agent.assess(
            project_id=project_id.strip(),
            incidents=records["incidents"],
            safety_trends=records["safety_trends"],
            compliance_score=compliance["overall_compliance_score"],
            total_violations=compliance["total_violations"],
        )

        # 3. Generate Targeted Recommendations ONLY from detected problems
        recommendations: List[str] = []
        if compliance["total_violations"] > 0:
            recommendations.append(
                f"Address {compliance['total_violations']} active safety violation(s) and apply corrective engineering/PPE controls."
            )
        if compliance["non_compliant_inspections"] > 0:
            recommendations.append(
                f"Perform corrective inspection and repair for {compliance['non_compliant_inspections']} non-compliant/failed inspection report(s)."
            )
        if compliance["non_compliant_logs"] > 0:
            recommendations.append(
                f"Improve safety-log compliance and mandate daily verified supervisor log entries to resolve {compliance['non_compliant_logs']} non-compliant log(s)."
            )
        urgent_incidents = insurance["high_severity_incidents"] + insurance["critical_incidents"]
        if urgent_incidents > 0:
            recommendations.append(
                f"Conduct immediate formal incident investigation for {urgent_incidents} high/critical severity event(s), implement corrective action, and assemble insurer claim documentation."
            )
        elif insurance["incident_count"] > 0:
            recommendations.append(
                f"Document all {insurance['incident_count']} reported incident(s) and review site safety protocol compliance."
            )

        if not recommendations:
            recommendations.append(
                "No major compliance or insurance risk indicators were identified from the submitted assessment data."
            )

        # 4. Project-Level Risk Scoring and Alert Threshold
        compliance_risk = 100.0 - compliance["overall_compliance_score"]
        total_risk = round((compliance_risk * 0.5) + (insurance["overall_insurance_risk"] * 0.5), 2)
        severity = self._severity(total_risk)

        alert = None
        if severity in {"HIGH", "CRITICAL"}:
            alert = {
                "alert_id": f"ALERT-{uuid.uuid4().hex[:8].upper()}",
                "project_id": project_id.strip(),
                "alert_type": "Project risk threshold exceeded",
                "severity": severity,
                "created_at": datetime.now().isoformat(),
            }
            try:
                self.db.save_project_alert(alert)
                self._notify(alert)
            except Exception:
                pass

        # Structured response conforming to Section 13 while retaining rich metadata
        compliance_summary = {
            **compliance,
            "overall_score": compliance["overall_compliance_score"],
            "level": compliance["compliance_level"],
            "regulatory_readiness": compliance["regulatory_readiness_score"],
        }
        insurance_summary = {
            **insurance,
            "overall_risk": insurance["overall_insurance_risk"],
        }

        return {
            "success": True,
            "project_id": project_id.strip(),
            "compliance": compliance_summary,
            "insurance": insurance_summary,
            "recommendations": recommendations,
            "total_risk_score": total_risk,
            "severity": severity,
            "alert": alert,
        }

    def _notify(self, alert: Dict) -> None:
        if self.notifier:
            self.notifier(alert)
        self.sms_service.send_alert_sms({
            "alert_id": alert["alert_id"],
            "timestamp": alert["created_at"],
            "violation_type": alert["alert_type"],
            "risk_level": alert["severity"],
            "worker_id": f"Project-{alert['project_id']}",
            "confidence": 1.0,
            "status": "New",
            "cooldown_key": f"PROJECT:{alert['project_id']}:{alert['severity']}",
        })

    @staticmethod
    def _severity(score: float) -> str:
        if score >= 75.0:
            return "CRITICAL"
        if score >= 50.0:
            return "HIGH"
        if score >= 25.0:
            return "MEDIUM"
        return "LOW"
