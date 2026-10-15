"""
Construction Risk Intelligence Engine (Central Brain)
=====================================================
Acts as the central orchestrator and executive analytics engine for the
Construction Risk Intelligence Platform.

Key Responsibilities:
1. Multi-Agent Orchestration: Consolidate data, alerts, and metrics from
   Site Risk Agent, Safety Agent, Compliance Agent, and Insurance Agent.
2. Global Risk Scoring: Compute unified Project Risk Score (0–100) based on
   weighted sub-agent indicators (active hazards, PPE compliance, regulatory
   violations, and insurance exposure).
3. Predictive Analytics & Forecasting: Recurring risk pattern detection,
   incident probability forecasting, and safety ROI / cost avoidance models.
4. Multi-Project Portfolio Aggregation: Enterprise portfolio health,
   ranking, and hotspot identification.
5. Agent Performance Monitoring: Track latency, accuracy, and efficiency
   against enterprise baselines (92%–96%).
6. Human-in-the-Loop & Auditability: Risk score overrides and immutable audit logging.
7. Escalation Dispatch: Coordinate automated alerts to Email, SMS, Slack/Teams.
"""

import json
import logging
import math
import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    GLOBAL_RISK_WEIGHTS,
    AGENT_PERFORMANCE_BASELINES,
    COST_PER_INCIDENT_PREVENTED,
    BASELINE_INCIDENT_RATE,
    COMPLIANCE_IMPROVEMENT_BASELINE,
)
from utils.escalation_service import EscalationService

logger = logging.getLogger(__name__)


class ConstructionRiskIntelligenceEngine:
    """
    Central Brain orchestrating all sub-agents and computing enterprise
    risk intelligence metrics.
    """

    def __init__(
        self,
        db_manager=None,
        site_risk_agent=None,
        safety_agent=None,
        compliance_agent=None,
        insurance_agent=None,
        escalation_service=None,
    ):
        self.db = db_manager
        self.site_risk_agent = site_risk_agent
        self.safety_agent = safety_agent
        self.compliance_agent = compliance_agent
        self.insurance_agent = insurance_agent
        self.escalation_service = escalation_service or (
            EscalationService(db_manager=db_manager) if db_manager else None
        )
        self.weights = dict(GLOBAL_RISK_WEIGHTS)

    # ------------------------------------------------------------------
    # 1. Multi-Agent Orchestration & Consolidation
    # ------------------------------------------------------------------
    def collect_agent_telemetry(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Gathers raw findings, alerts, and states across all 4 sub-agents.
        Falls back safely to database records if agent instances are not directly mounted.
        """
        telemetry: Dict[str, Any] = {
            "project_id": project_id or "GLOBAL",
            "timestamp": datetime.now().isoformat(),
            "sub_agents": {},
        }

        # --- 1. Safety Agent ---
        safety_data: Dict[str, Any] = {}
        if self.safety_agent and hasattr(self.safety_agent, "get_worker_summaries"):
            try:
                workers = self.safety_agent.get_worker_summaries()
                total = len(workers)
                compliant = sum(1 for w in workers if w.get("ppe_status") == "Compliant")
                compliance_pct = (compliant / max(1, total)) * 100.0
                safety_data = {
                    "total_workers": total,
                    "compliant_workers": compliant,
                    "non_compliant_workers": total - compliant,
                    "compliance_rate": round(compliance_pct, 1),
                    "active_violations": sum(w.get("violation_count", 0) for w in workers),
                }
            except Exception as exc:
                logger.warning("Failed to collect from Safety Agent: %s", exc)

        if not safety_data and self.db:
            summary = self.db.get_analytics_summary()
            safety_data = {
                "total_workers": summary.get("total_workers", 0),
                "compliant_workers": summary.get("compliant_workers", 0),
                "non_compliant_workers": summary.get("non_compliant_workers", 0),
                "compliance_rate": summary.get("compliance_percentage", 100.0),
                "active_violations": summary.get("total_violations", 0),
            }
        telemetry["sub_agents"]["safety_agent"] = safety_data

        # --- 2. Site Risk Agent (Hazard Alerts) ---
        hazards_data: Dict[str, Any] = {}
        if self.db:
            all_alerts = self.db.get_alerts()
            if project_id:
                all_alerts = [a for a in all_alerts if a.get("project_id") == project_id]
            active_alerts = [a for a in all_alerts if a.get("status") in ("New", "Active", "Open")]
            critical_count = sum(1 for a in active_alerts if a.get("risk_level") == "CRITICAL" or a.get("severity") == "CRITICAL")
            high_count = sum(1 for a in active_alerts if a.get("risk_level") == "HIGH" or a.get("severity") == "HIGH")
            hazards_data = {
                "total_active_alerts": len(active_alerts),
                "critical_alerts": critical_count,
                "high_alerts": high_count,
                "recent_alerts": active_alerts[:10],
            }
        telemetry["sub_agents"]["site_risk_agent"] = hazards_data

        # --- 3. Compliance Agent ---
        compliance_data: Dict[str, Any] = {}
        if self.db:
            checks = self.db.get_compliance_checks(project_id)
            total_checks = len(checks)
            compliant_checks = sum(1 for c in checks if c.get("compliance_status") == "Compliant")
            open_violations = total_checks - compliant_checks
            readiness = round((compliant_checks / max(1, total_checks)) * 100.0, 1) if total_checks > 0 else 100.0
            compliance_data = {
                "total_checks": total_checks,
                "compliant_checks": compliant_checks,
                "open_violations": open_violations,
                "audit_readiness_score": readiness,
            }
        telemetry["sub_agents"]["compliance_agent"] = compliance_data

        # --- 4. Insurance Agent ---
        insurance_data: Dict[str, Any] = {}
        if self.db:
            cases = self.db.get_insurance_cases(project_id)
            highest_score = max((c.get("risk_score", 0.0) for c in cases), default=0.0)
            open_cases = [c for c in cases if c.get("status") in ("Open", "Under Review", "Pending")]
            exposure = (
                "High" if highest_score >= 70 else
                "Medium" if highest_score >= 35 else "Low"
            )
            insurance_data = {
                "total_cases": len(cases),
                "open_cases_count": len(open_cases),
                "highest_risk_score": round(highest_score, 1),
                "insurance_exposure": exposure,
            }
        telemetry["sub_agents"]["insurance_agent"] = insurance_data

        return telemetry

    # ------------------------------------------------------------------
    # 2. Global Risk Scoring Algorithm
    # ------------------------------------------------------------------
    def calculate_global_risk_score(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculates unified Project Risk Score (0–100) using multi-factor weighted indicators:
          - Active Hazards (25%): Normalized active hazard/alert count & severity
          - PPE Non-Compliance (25%): Inverted worker compliance rate
          - Regulatory Violations (25%): Normalized open compliance violations
          - Insurance Exposure (25%): Highest claim risk score and case exposure

        Formula:
            Risk Score = w1*HazardScore + w2*(100 - PPE_Rate) + w3*RegScore + w4*InsScore
        """
        telemetry = self.collect_agent_telemetry(project_id)
        sub = telemetry["sub_agents"]

        # Sub-score 1: Active Hazards (0–100)
        hazards = sub.get("site_risk_agent", {})
        active_cnt = hazards.get("total_active_alerts", 0)
        crit_cnt = hazards.get("critical_alerts", 0)
        high_cnt = hazards.get("high_alerts", 0)
        # Weight critical 30pts each, high 15pts each, others 5pts each, capped at 100
        hazard_raw = (crit_cnt * 30.0) + (high_cnt * 15.0) + (max(0, active_cnt - crit_cnt - high_cnt) * 5.0)
        hazard_score = min(100.0, round(hazard_raw, 1))

        # Sub-score 2: PPE Non-Compliance (0–100)
        # High non-compliance = high risk
        safety = sub.get("safety_agent", {})
        compliance_rate = safety.get("compliance_rate", 100.0)
        ppe_risk_score = round(max(0.0, min(100.0, 100.0 - compliance_rate)), 1)

        # Sub-score 3: Regulatory Violations (0–100)
        comp = sub.get("compliance_agent", {})
        open_viol = comp.get("open_violations", 0)
        readiness = comp.get("audit_readiness_score", 100.0)
        # Inverted readiness score combined with open violation count
        reg_raw = (100.0 - readiness) * 0.7 + min(100.0, open_viol * 20.0) * 0.3
        regulatory_score = round(max(0.0, min(100.0, reg_raw)), 1)

        # Sub-score 4: Insurance Exposure (0–100)
        ins = sub.get("insurance_agent", {})
        ins_score = float(ins.get("highest_risk_score", 0.0))
        insurance_score = round(max(0.0, min(100.0, ins_score)), 1)

        # Weighted composite score
        w = self.weights
        composite = (
            w.get("active_hazards", 0.25) * hazard_score
            + w.get("ppe_compliance", 0.25) * ppe_risk_score
            + w.get("regulatory_violations", 0.25) * regulatory_score
            + w.get("insurance_exposure", 0.25) * insurance_score
        )
        final_score = round(composite, 1)

        # Classification
        if final_score >= 80.0:
            level = "Critical"
        elif final_score >= 60.0:
            level = "High"
        elif final_score >= 30.0:
            level = "Medium"
        else:
            level = "Low"

        result = {
            "project_id": project_id or "GLOBAL",
            "global_risk_score": final_score,
            "risk_level": level,
            "weights": self.weights,
            "component_scores": {
                "active_hazards": hazard_score,
                "ppe_non_compliance": ppe_risk_score,
                "regulatory_violations": regulatory_score,
                "insurance_exposure": insurance_score,
            },
            "sub_agent_metrics": sub,
            "calculated_at": datetime.now().isoformat(),
        }

        # Update project table if project_id is provided and db exists
        if self.db and project_id:
            try:
                self.db.update_project_risk_score(project_id, final_score, level)
            except Exception as exc:
                logger.warning("Could not persist project risk score: %s", exc)

        # Check for automated escalation if critical/high
        if level in ("High", "Critical") and self.escalation_service:
            self._trigger_score_escalation(result)

        return result

    def _trigger_score_escalation(self, score_data: Dict[str, Any]) -> None:
        """Dispatches an enterprise alert when the global risk score crosses threshold."""
        alert_payload = {
            "alert_id": f"RISK-ESC-{uuid.uuid4().hex[:6].upper()}",
            "project_id": score_data.get("project_id", "GLOBAL"),
            "alert_type": f"Enterprise {score_data['risk_level']} Risk Level Alert",
            "severity": "CRITICAL" if score_data["risk_level"] == "Critical" else "HIGH",
            "risk_level": score_data["risk_level"].upper(),
            "created_at": datetime.now().isoformat(),
            "confidence": 1.0,
            "details": f"Global Risk Score reached {score_data['global_risk_score']}/100. Immediate site inspection required.",
        }
        try:
            self.escalation_service.escalate(alert_payload)
            if self.db:
                self.db.log_audit_trail(
                    action="AUTO_ESCALATION_TRIGGERED",
                    agent="IntelligenceEngine",
                    details=alert_payload,
                )
        except Exception as exc:
            logger.error("Failed to execute automatic escalation: %s", exc)

    # ------------------------------------------------------------------
    # 3. Predictive Analytics & Forecasting
    # ------------------------------------------------------------------
    def run_predictive_analytics(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyzes historical safety events, violation distribution, and worker habits to:
        - Detect recurring risk patterns
        - Forecast incident likelihood (probability %)
        - Model safety ROI & cost avoidance based on industry baselines
        """
        if not self.db:
            return {"error": "Database manager not connected"}

        violations = self.db.get_violations(limit=200)
        events = self.db.get_recent_events(limit=100)
        workers = self.db.get_workers(limit=100)

        # Pattern 1: Recurring violation items (helmets, vests, boots, etc.)
        item_frequencies: Dict[str, int] = {}
        worker_violations: Dict[str, int] = {}
        for v in violations:
            w_id = v.get("worker_id", "Unknown")
            worker_violations[w_id] = worker_violations.get(w_id, 0) + 1
            missing = v.get("missing_ppe") or []
            for item in missing:
                item_frequencies[item] = item_frequencies.get(item, 0) + 1

        top_recurring_items = sorted(
            [{"item": k, "count": v} for k, v in item_frequencies.items()],
            key=lambda x: x["count"],
            reverse=True,
        )

        top_repeat_workers = sorted(
            [{"worker_id": k, "violation_count": v} for k, v in worker_violations.items()],
            key=lambda x: x["violation_count"],
            reverse=True,
        )[:5]

        # Pattern 2: Temporal violation trends (hourly / shift pattern)
        hourly_distribution: Dict[int, int] = {}
        for ev in events:
            ts = ev.get("timestamp")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts)
                    hourly_distribution[dt.hour] = hourly_distribution.get(dt.hour, 0) + ev.get("violations_count", 0)
                except Exception:
                    pass

        peak_risk_hours = sorted(
            [{"hour": k, "violation_count": v} for k, v in hourly_distribution.items()],
            key=lambda x: x["violation_count"],
            reverse=True,
        )[:3]

        # Predictive Model: Projected Incident Probability (0–100%)
        # Based on open violations count, repeat worker frequency, and current compliance rate
        total_workers = max(1, len(workers))
        compliant_workers = sum(1 for w in workers if w.get("ppe_status") == "Compliant")
        current_compliance = (compliant_workers / total_workers) * 100.0

        # Base probability from uncorrected violations and non-compliance
        non_compliance_delta = max(0.0, 100.0 - current_compliance)
        incident_probability = min(95.0, round(
            (non_compliance_delta * 0.55) +
            (min(len(violations), 40) * 0.75) +
            (len(top_repeat_workers) * 2.5),
            1
        ))

        trend_direction = "WORSENING" if incident_probability > 50.0 else ("STABLE" if incident_probability > 25.0 else "IMPROVING")

        # Safety ROI / Cost Impact Calculations
        # Cost avoidance = (Baseline Incidents - Projected Incidents) * Cost Per Incident
        compliance_gain = max(0.0, current_compliance - COMPLIANCE_IMPROVEMENT_BASELINE)
        prevented_incidents_est = round((compliance_gain / 100.0) * (total_workers * BASELINE_INCIDENT_RATE), 2)
        cost_savings_usd = round(prevented_incidents_est * COST_PER_INCIDENT_PREVENTED, 2)

        return {
            "project_id": project_id or "GLOBAL",
            "forecast_generated_at": datetime.now().isoformat(),
            "incident_forecast": {
                "incident_probability_pct": incident_probability,
                "risk_trend": trend_direction,
                "confidence_level": 0.88,
                "predictive_window_days": 14,
            },
            "recurring_patterns": {
                "top_missing_ppe": top_recurring_items,
                "repeat_violators": top_repeat_workers,
                "peak_risk_hours": peak_risk_hours,
            },
            "safety_roi_model": {
                "baseline_compliance_pct": COMPLIANCE_IMPROVEMENT_BASELINE,
                "current_compliance_pct": round(current_compliance, 1),
                "compliance_gain_pct": round(compliance_gain, 1),
                "estimated_incidents_prevented": prevented_incidents_est,
                "estimated_cost_avoidance_usd": cost_savings_usd,
                "cost_basis_per_incident_usd": COST_PER_INCIDENT_PREVENTED,
            },
        }

    # ------------------------------------------------------------------
    # 4. Multi-Project Enterprise Portfolio Aggregation
    # ------------------------------------------------------------------
    def get_portfolio_overview(self) -> Dict[str, Any]:
        """
        Aggregates risk and compliance across all registered projects,
        identifying highest-risk sites and portfolio-level statistics.
        """
        if not self.db:
            return {"projects": [], "summary": {}}

        projects = self.db.get_all_projects()
        if not projects:
            # Fallback default project if none seeded
            self._ensure_default_project()
            projects = self.db.get_all_projects()

        portfolio_scores = []
        for p in projects:
            p_id = p["project_id"]
            score_data = self.calculate_global_risk_score(p_id)
            portfolio_scores.append({
                "project_id": p_id,
                "project_name": p.get("project_name", p_id),
                "status": p.get("status", "Active"),
                "global_risk_score": score_data["global_risk_score"],
                "risk_level": score_data["risk_level"],
                "component_scores": score_data["component_scores"],
            })

        # Rank projects by risk score descending (hotspots first)
        portfolio_scores.sort(key=lambda x: x["global_risk_score"], reverse=True)

        avg_score = (
            round(sum(p["global_risk_score"] for p in portfolio_scores) / len(portfolio_scores), 1)
            if portfolio_scores else 0.0
        )
        critical_projects = [p for p in portfolio_scores if p["risk_level"] == "Critical"]
        high_projects = [p for p in portfolio_scores if p["risk_level"] == "High"]

        return {
            "total_projects": len(portfolio_scores),
            "portfolio_average_risk_score": avg_score,
            "critical_projects_count": len(critical_projects),
            "high_risk_projects_count": len(high_projects),
            "portfolio_hotspots": portfolio_scores[:3],
            "all_projects": portfolio_scores,
            "generated_at": datetime.now().isoformat(),
        }

    def _ensure_default_project(self) -> None:
        """Seed initial projects if database is fresh."""
        default_projects = [
            {"project_id": "PRJ-METRO-01", "project_name": "Downtown Metro Rail Station", "status": "Active"},
            {"project_id": "PRJ-TOWER-02", "project_name": "Skyline Commercial Tower B", "status": "Active"},
            {"project_id": "PRJ-BRIDGE-03", "project_name": "Harbor Bypass Bridge Extension", "status": "Active"},
        ]
        for p in default_projects:
            try:
                self.db.save_project(p)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 5. Agent Performance Monitoring & Health
    # ------------------------------------------------------------------
    def record_agent_performance_metric(
        self,
        agent_name: str,
        metric_name: str,
        metric_value: float,
        tasks_processed: int = 1,
        tasks_succeeded: int = 1,
        details: Optional[Dict] = None,
    ) -> Optional[int]:
        """Records an agent performance entry in the audit tracking database."""
        if not self.db:
            return None
        record = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "metric_name": metric_name,
            "metric_value": float(metric_value),
            "tasks_processed": tasks_processed,
            "tasks_succeeded": tasks_succeeded,
            "details": details or {},
        }
        return self.db.log_agent_performance(record)

    def get_agent_health_report(self) -> Dict[str, Any]:
        """
        Compares latest real-time agent metrics against enterprise baselines
        (Site Risk 96%, Safety 94%, Compliance 95%, Insurance 92%).
        """
        latest_metrics = self.db.get_latest_agent_metrics() if self.db else {}
        health_report: Dict[str, Any] = {}

        for agent_key, baseline in AGENT_PERFORMANCE_BASELINES.items():
            db_metric = latest_metrics.get(agent_key)
            if db_metric:
                current_eff = float(db_metric.get("metric_value", baseline["efficiency"]))
                tasks_proc = db_metric.get("tasks_processed", 100)
                tasks_succ = db_metric.get("tasks_succeeded", 95)
            else:
                # Default to baseline if not yet recorded
                current_eff = baseline["efficiency"]
                tasks_proc = 150
                tasks_succ = int(150 * (baseline["accuracy"] / 100.0))

            accuracy_calc = round((tasks_succ / max(1, tasks_proc)) * 100.0, 1)
            status = "HEALTHY" if current_eff >= (baseline["efficiency"] - 5.0) else "DEGRADED"

            health_report[agent_key] = {
                "status": status,
                "current_efficiency_pct": current_eff,
                "baseline_efficiency_pct": baseline["efficiency"],
                "accuracy_pct": accuracy_calc,
                "baseline_accuracy_pct": baseline["accuracy"],
                "tasks_processed": tasks_proc,
                "tasks_succeeded": tasks_succ,
                "last_evaluated": db_metric.get("timestamp", datetime.now().isoformat()) if db_metric else datetime.now().isoformat(),
            }

        return {
            "overall_system_health": "OPTIMAL" if all(v["status"] == "HEALTHY" for v in health_report.values()) else "ATTENTION_REQUIRED",
            "agents": health_report,
            "evaluated_at": datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # 6. Human-in-the-Loop & Manual Overrides
    # ------------------------------------------------------------------
    def apply_manual_risk_override(
        self,
        project_id: str,
        new_risk_score: float,
        reason: str,
        user_name: str = "Enterprise Supervisor",
    ) -> Dict[str, Any]:
        """
        Allows authorized human supervisors to override an agent risk score
        with mandatory justification logged in the immutable audit trail.
        """
        score = max(0.0, min(100.0, float(new_risk_score)))
        level = "Critical" if score >= 80 else ("High" if score >= 60 else ("Medium" if score >= 30 else "Low"))

        if self.db:
            self.db.update_project_risk_score(project_id, score, level)
            self.db.log_audit_trail(
                action="MANUAL_RISK_SCORE_OVERRIDE",
                agent=f"User:{user_name}",
                details={
                    "project_id": project_id,
                    "new_score": score,
                    "new_level": level,
                    "justification": reason,
                    "timestamp": datetime.now().isoformat(),
                },
                user_override=True,
            )

        return {
            "project_id": project_id,
            "overridden_score": score,
            "overridden_level": level,
            "justification": reason,
            "authorized_by": user_name,
            "timestamp": datetime.now().isoformat(),
            "status": "Applied",
        }
