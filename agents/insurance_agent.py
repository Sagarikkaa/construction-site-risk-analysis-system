"""Transparent, deterministic rule-based insurance exposure assessment."""

import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from database.db_manager import SafetyDBManager


class InsuranceAgent:
    """Calculate claim and insurance risk from incidents, violations, and compliance score."""

    SEVERITY_SCORES = {
        "low": 10.0,
        "medium": 25.0,
        "high": 50.0,
        "critical": 75.0,
    }

    SEVERITY_KEYWORDS = {
        "critical": ("fatal", "fatality", "major fire", "critical injury", "collapse", "explosion"),
        "high": ("serious injury", "severe", "fracture", "hospitalization", "fall from height", "collision"),
        "medium": ("equipment damage", "near miss", "property damage", "spill", "machinery"),
        "low": ("minor injury", "first aid", "minor scrape", "cut", "bruise"),
    }

    def __init__(self, db_manager: Optional[SafetyDBManager] = None):
        self.db = db_manager or SafetyDBManager()

    def assess(
        self,
        project_id: str,
        incidents: Iterable[Dict] = (),
        safety_trends: Iterable[Dict] = (),
        compliance_score: float = 100.0,
        total_violations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Calculate and persist a deterministic insurance risk assessment."""
        incident_records = list(incidents)
        if total_violations is None:
            total_violations = self._total_violations(safety_trends)

        # 1. Incident Risk & Severity Counting
        low_count = 0
        med_count = 0
        high_count = 0
        crit_count = 0
        incident_points: List[float] = []
        claim_factors: List[str] = []

        for item in incident_records:
            if not isinstance(item, dict):
                continue
            sev = self._severity(item)
            points = self.SEVERITY_SCORES.get(sev, 10.0)
            incident_points.append(points)
            if sev == "critical":
                crit_count += 1
            elif sev == "high":
                high_count += 1
            elif sev == "medium":
                med_count += 1
            else:
                low_count += 1
            desc = item.get("description", item.get("finding", item.get("type", "Incident")))
            claim_factors.append(f"{desc} (Severity: {sev.title()}, Points: {points})")

        if incident_points:
            incident_risk = round(sum(incident_points) / len(incident_points), 2)
        else:
            incident_risk = 0.0

        # 2. Violation Risk
        violation_risk = self._violation_risk(total_violations)
        if total_violations > 0:
            claim_factors.append(f"{total_violations} recorded violation(s) generating {violation_risk} risk points")

        # 3. Compliance Risk
        compliance_risk = self._compliance_risk(compliance_score)
        if compliance_risk > 0:
            claim_factors.append(
                f"Compliance score of {compliance_score} adds {compliance_risk} risk points to insurance liability"
            )

        # 4. Overall Weighted Insurance Risk
        overall_insurance_risk = round(
            (incident_risk * 0.50) + (violation_risk * 0.25) + (compliance_risk * 0.25),
            2,
        )

        insurance_risk_level = self._risk_level(overall_insurance_risk)
        claim_risk = insurance_risk_level
        insurance_exposure = self._exposure(insurance_risk_level)

        # Recommendations based ONLY on detected problems
        recommended_actions = self._generate_recommendations(
            total_violations=total_violations,
            high_or_critical_incidents=(high_count + crit_count),
            incident_count=len(incident_records),
            compliance_risk=compliance_risk,
        )

        # Explainability text
        explanation = (
            f"Incident Risk: {incident_risk} (50% weight), "
            f"Violation Risk: {violation_risk} (25% weight), "
            f"Compliance Risk: {compliance_risk} (25% weight). "
            f"Overall Insurance Risk: {overall_insurance_risk}/100 ({insurance_risk_level})."
        )

        # Database case persistence with valid timestamp
        now_iso = datetime.now().isoformat()
        case = {
            "case_id": f"CASE-{uuid.uuid4().hex[:8].upper()}",
            "project_id": project_id,
            "claim_type": self._claim_type(incident_records),
            "risk_score": overall_insurance_risk,
            "status": "Open" if overall_insurance_risk >= 20.0 else "Monitoring",
            "timestamp": now_iso,
        }
        try:
            self.db.save_insurance_case(case)
        except Exception as exc:
            # Let caller handle or raise database errors appropriately
            raise RuntimeError(f"Failed to persist insurance case to database: {exc}") from exc

        return {
            "project_id": project_id,
            "case": case,
            "incident_count": len(incident_records),
            "low_severity_incidents": low_count,
            "medium_severity_incidents": med_count,
            "high_severity_incidents": high_count,
            "critical_incidents": crit_count,
            "incident_risk": incident_risk,
            "violation_risk": violation_risk,
            "compliance_risk": compliance_risk,
            "overall_insurance_risk": overall_insurance_risk,
            "insurance_risk_score": overall_insurance_risk,
            "risk_score": overall_insurance_risk,
            "insurance_risk_level": insurance_risk_level,
            "risk_level": insurance_risk_level,
            "claim_risk": claim_risk,
            "insurance_exposure": insurance_exposure,
            "claim_factors": claim_factors,
            "recommended_actions": recommended_actions,
            "explanation": explanation,
            "rule_based": True,
        }

    @classmethod
    def _severity(cls, incident: Dict) -> str:
        raw = str(incident.get("severity", "")).strip().lower()
        if raw in cls.SEVERITY_SCORES:
            return raw
        text = " ".join(str(value).lower() for value in incident.values())
        for severity, keywords in cls.SEVERITY_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                return severity
        return "low"

    @staticmethod
    def _violation_risk(total_violations: int) -> float:
        if total_violations == 0:
            return 0.0
        if total_violations <= 2:
            return 10.0
        if total_violations <= 5:
            return 25.0
        if total_violations <= 10:
            return 50.0
        return 75.0

    @staticmethod
    def _compliance_risk(compliance_score: float) -> float:
        if compliance_score >= 90.0:
            return 0.0
        if compliance_score >= 75.0:
            return 10.0
        if compliance_score >= 50.0:
            return 25.0
        if compliance_score >= 25.0:
            return 50.0
        return 75.0

    @staticmethod
    def _risk_level(score: float) -> str:
        if score >= 80.0:
            return "Critical"
        if score >= 60.0:
            return "High"
        if score >= 40.0:
            return "Moderate"
        if score >= 20.0:
            return "Low"
        return "Very Low"

    @staticmethod
    def _exposure(risk_level: str) -> str:
        return {
            "Very Low": "Minimal",
            "Low": "Limited",
            "Moderate": "Moderate",
            "High": "Significant",
            "Critical": "Severe",
        }.get(risk_level, "Minimal")

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
    def _claim_type(incidents: Iterable[Dict]) -> str:
        first = next(iter(incidents), {})
        return str(first.get("claim_type", first.get("type", "General construction risk assessment")))

    @classmethod
    def _generate_recommendations(
        cls,
        total_violations: int,
        high_or_critical_incidents: int,
        incident_count: int,
        compliance_risk: float,
    ) -> List[str]:
        actions: List[str] = []
        if high_or_critical_incidents > 0:
            actions.append(
                "Immediately initiate root-cause investigation for high/critical incident(s), preserve evidence, and notify insurance carrier."
            )
        elif incident_count > 0:
            actions.append(
                "Document all minor/medium reported incidents and conduct supervisor safety debriefing."
            )
        if total_violations > 0:
            actions.append(
                "Execute corrective actions for active safety violations to prevent liability escalation."
            )
        if compliance_risk >= 25.0:
            actions.append(
                "Rectify regulatory compliance deficits to mitigate potential non-compliance insurance denial."
            )
        return actions
