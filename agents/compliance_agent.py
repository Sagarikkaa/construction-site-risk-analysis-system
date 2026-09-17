"""Transparent, deterministic rule-based construction compliance assessment."""

import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from database.db_manager import SafetyDBManager


class ComplianceAgent:
    """Score inspections, safety logs, and violation trends deterministically."""

    DEFAULT_RULES = {
        "OSHA PPE": ("helmet", "hardhat", "safety vest", "ppe"),
        "Building Code": ("scaffold", "guardrail", "structural", "building code"),
        "Environmental Regulation": ("spill", "waste", "dust", "environment"),
    }

    STATUS_SCORES = {
        "compliant": 100.0,
        "non-compliant": 60.0,
        "failed": 40.0,
        "violation": 30.0,
    }

    def __init__(self, db_manager: Optional[SafetyDBManager] = None, rules: Optional[Dict] = None):
        self.db = db_manager or SafetyDBManager()
        self.rules = rules or self.DEFAULT_RULES

    def evaluate(
        self,
        project_id: str,
        inspection_reports: Iterable[Dict] = (),
        safety_logs: Iterable[Dict] = (),
        safety_violation_trends: Iterable[Dict] = (),
    ) -> Dict[str, Any]:
        """Calculate the compliance score strictly from submitted data and persist checks."""
        inspections = list(inspection_reports)
        logs = list(safety_logs)
        trends = list(safety_violation_trends)

        # 1. Inspection Score
        inspection_scores: List[float] = []
        compliant_inspections = 0
        non_compliant_inspections = 0
        compliance_findings: List[str] = []

        for item in inspections:
            if not isinstance(item, dict):
                continue
            status = self._normalize_status(item)
            score = self.STATUS_SCORES.get(status, 60.0)
            inspection_scores.append(score)
            finding = item.get("finding", item.get("description", item.get("notes", "")))
            if finding:
                compliance_findings.append(f"Inspection: {finding} ({status})")
            if status == "compliant":
                compliant_inspections += 1
            else:
                non_compliant_inspections += 1

        if inspection_scores:
            inspection_score = round(sum(inspection_scores) / len(inspection_scores), 2)
        else:
            inspection_score = 50.0

        # 2. Safety Log Score
        log_scores: List[float] = []
        compliant_logs = 0
        non_compliant_logs = 0

        for item in logs:
            if not isinstance(item, dict):
                continue
            status = self._normalize_status(item)
            score = self.STATUS_SCORES.get(status, 60.0)
            log_scores.append(score)
            finding = item.get("finding", item.get("description", item.get("notes", "")))
            if finding:
                compliance_findings.append(f"Safety Log: {finding} ({status})")
            if status == "compliant":
                compliant_logs += 1
            else:
                non_compliant_logs += 1

        if log_scores:
            safety_log_score = round(sum(log_scores) / len(log_scores), 2)
        else:
            safety_log_score = 50.0

        # 3. Violation Score
        total_violations = self._total_violations(trends)
        violation_score = self._violation_score(total_violations)

        # 4. Overall Weighted Compliance Score
        overall_compliance_score = round(
            (inspection_score * 0.40) + (safety_log_score * 0.30) + (violation_score * 0.30),
            2,
        )
        compliance_level = self._level(overall_compliance_score)
        regulatory_readiness_score = overall_compliance_score

        # Corrective actions
        corrective_actions: List[str] = []
        if total_violations > 0:
            corrective_actions.append(
                f"Address and resolve {total_violations} recorded safety violation trend(s) immediately."
            )
        if non_compliant_inspections > 0:
            corrective_actions.append(
                f"Conduct re-inspections for {non_compliant_inspections} non-compliant/failed inspection item(s)."
            )
        if non_compliant_logs > 0:
            corrective_actions.append(
                f"Review and rectify {non_compliant_logs} non-compliant safety log item(s) to enforce site protocols."
            )
        if not corrective_actions:
            corrective_actions.append(
                "Maintain current high safety compliance controls and daily audit logs."
            )

        # Explanation / Breakdown text
        explanation = (
            f"Inspection Score: {inspection_score} (40% weight), "
            f"Safety Log Score: {safety_log_score} (30% weight), "
            f"Violation Score: {violation_score} (30% weight). "
            f"Overall Compliance: {overall_compliance_score}/100 ({compliance_level})."
        )

        # Save checks to database
        checks: List[Dict] = []
        for regulation_name in self.rules.keys():
            status = "Compliant" if overall_compliance_score >= 75.0 else "Non-Compliant"
            check = {
                "compliance_id": f"COMP-{uuid.uuid4().hex[:8].upper()}",
                "project_id": project_id,
                "regulation_name": regulation_name,
                "compliance_status": status,
                "checked_at": datetime.now().isoformat(),
            }
            self.db.save_compliance_check(check)
            checks.append(check)

        return {
            "project_id": project_id,
            "checks": checks,
            "inspection_score": inspection_score,
            "safety_log_score": safety_log_score,
            "violation_score": violation_score,
            "overall_compliance_score": overall_compliance_score,
            "compliance_score": overall_compliance_score,
            "compliance_level": compliance_level,
            "regulatory_readiness_score": regulatory_readiness_score,
            "audit_readiness_score": regulatory_readiness_score,
            "total_inspections": len(inspections),
            "compliant_inspections": compliant_inspections,
            "non_compliant_inspections": non_compliant_inspections,
            "total_safety_logs": len(logs),
            "compliant_logs": compliant_logs,
            "non_compliant_logs": non_compliant_logs,
            "total_violations": total_violations,
            "open_violations": total_violations,
            "compliance_findings": compliance_findings,
            "corrective_actions": corrective_actions,
            "explanation": explanation,
        }

    @staticmethod
    def _normalize_status(record: Dict) -> str:
        """Normalize status into compliant, non-compliant, failed, violation."""
        raw = str(record.get("status", record.get("compliance_status", ""))).strip().lower()
        if not raw:
            return "non-compliant"
        if any(w in raw for w in ("violation", "breach")):
            return "violation"
        if any(w in raw for w in ("fail", "failed", "critical")):
            return "failed"
        if any(w in raw for w in ("non-compliant", "noncompliant", "unsafe", "incomplete", "warning")):
            return "non-compliant"
        if any(w in raw for w in ("compliant", "pass", "passed", "ok", "verified", "safe")):
            return "compliant"
        return "non-compliant"

    @staticmethod
    def _total_violations(trends: Iterable[Dict]) -> int:
        total = 0
        for trend in trends:
            if isinstance(trend, dict):
                try:
                    total += max(0, int(float(trend.get("violations", trend.get("count", 0)) or 0)))
                except (TypeError, ValueError):
                    continue
        return total

    @staticmethod
    def _violation_score(violations: int) -> float:
        if violations == 0:
            return 100.0
        if violations <= 2:
            return 90.0
        if violations <= 5:
            return 75.0
        if violations <= 10:
            return 50.0
        return 25.0

    @staticmethod
    def _level(score: float) -> str:
        if score >= 90:
            return "Excellent"
        if score >= 75:
            return "Good"
        if score >= 50:
            return "Moderate"
        if score >= 25:
            return "Poor"
        return "Critical"
