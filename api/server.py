"""
Safety Intelligence Platform REST API
======================================
Provides backend endpoints for safety analysis, PPE detection, alert tracking,
analytics metrics, and worker monitoring records.

Endpoints:
    POST  /api/safety/analyze   - Complete safety analysis on image/video
    POST  /api/safety/detect    - Raw YOLO detection + worker-PPE association
    GET   /api/safety/alerts    - Retrieve safety alerts (supports ?status=New/Acknowledged/Resolved)
    PATCH /api/safety/alerts/{id} - Update alert status
    GET   /api/safety/analytics - Aggregate analytics and chart metrics
    GET   /api/safety/workers   - Active worker safety monitoring records
"""

import os
import sys
from typing import Dict, List, Optional
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.safety_agent import SafetyAgent
from agents.reporting_agent import ReportingAgent
from database.db_manager import SafetyDBManager
from risk.construction_risk_engine import ConstructionRiskEngine
from risk.intelligence_engine import ConstructionRiskIntelligenceEngine
from utils.config import DEFAULT_CONFIDENCE_THRESHOLD, ALERT_STATUSES
from utils.sms_service import SMSService
from utils.escalation_service import EscalationService

try:
    from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Body
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, PlainTextResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


# Shared instances
safety_agent = SafetyAgent()
db_manager = SafetyDBManager()
sms_service = SMSService(db_manager=db_manager)
risk_engine = ConstructionRiskEngine(db_manager=db_manager)
escalation_service = EscalationService(db_manager=db_manager, sms_service=sms_service)
intelligence_engine = ConstructionRiskIntelligenceEngine(
    db_manager=db_manager,
    safety_agent=safety_agent,
    escalation_service=escalation_service,
)
reporting_agent = ReportingAgent(
    db_manager=db_manager,
    intelligence_engine=intelligence_engine,
)


# ------------------------------------------------------------------
# FastAPI Server Definition
# ------------------------------------------------------------------
if HAS_FASTAPI:
    app = FastAPI(
        title="Construction Risk Intelligence Platform API",
        description="REST API for Milestone 2: Safety Intelligence & Worker Protection",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class AlertStatusUpdate(BaseModel):
        status: str

    class AnalysisRequest(BaseModel):
        image_path: Optional[str] = None
        confidence_threshold: Optional[float] = DEFAULT_CONFIDENCE_THRESHOLD

    class SendSMSRequest(BaseModel):
        worker_id: str
        phone_number: str
        risk_level: str
        message: str

    class ProjectAssessmentRequest(BaseModel):
        project_id: str
        inspection_reports: List[Dict] = []
        safety_logs: List[Dict] = []
        incidents: List[Dict] = []
        safety_trends: List[Dict] = []

    class ProjectCreateRequest(BaseModel):
        project_id: str
        project_name: Optional[str] = None
        status: Optional[str] = "Active"
        metadata: Optional[Dict] = None

    class RiskOverrideRequest(BaseModel):
        project_id: str
        new_risk_score: float
        reason: str
        user_name: Optional[str] = "Enterprise Supervisor"

    class ReportGenerateRequest(BaseModel):
        project_id: Optional[str] = "PRJ-METRO-01"
        report_type: str = "daily_operational"
        export_format: Optional[str] = "json"

    @app.get("/")
    def root():
        return {
            "platform": "Construction Risk Intelligence Platform",
            "milestone": "Milestones 1-4: Enterprise Risk Intelligence & Reporting",
            "status": "Online",
            "endpoints": [
                "/api/safety/analyze",
                "/api/safety/detect",
                "/api/safety/alerts",
                "/api/safety/analytics",
                "/api/safety/workers",
                "/api/safety/sms-logs",
                "/api/v1/enterprise/dashboard",
                "/api/v1/enterprise/projects",
                "/api/v1/risk/global-score",
                "/api/v1/risk/override",
                "/api/v1/risk/predictive-forecast",
                "/api/v1/reports/generate",
                "/api/v1/reports",
                "/api/v1/system/agent-health",
                "/api/v1/audit/trail",
            ]
        }

    @app.post("/api/safety/analyze")
    async def analyze(
        file: Optional[UploadFile] = File(None),
        image_path: Optional[str] = Query(None),
        confidence_threshold: float = Query(DEFAULT_CONFIDENCE_THRESHOLD),
    ):
        """Run full Safety Agent analysis on an uploaded image or local file path."""
        try:
            if file:
                contents = await file.read()
                report = safety_agent.analyze_uploaded_image(
                    contents, filename=file.filename, conf_threshold=confidence_threshold
                )
            elif image_path:
                report = safety_agent.analyze_image(image_path, conf_threshold=confidence_threshold)
            else:
                raise HTTPException(status_code=400, detail="Provide an uploaded image file or image_path query parameter.")

            # Sanitize report for JSON serialization (exclude raw numpy arrays)
            return _sanitize_report(report)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/safety/detect")
    async def detect(
        file: Optional[UploadFile] = File(None),
        image_path: Optional[str] = Query(None),
        confidence_threshold: float = Query(DEFAULT_CONFIDENCE_THRESHOLD),
    ):
        """Run raw detection and PPE spatial association."""
        try:
            if file:
                contents = await file.read()
                report = safety_agent.analyze_uploaded_image(
                    contents, filename=file.filename, conf_threshold=confidence_threshold
                )
            elif image_path:
                report = safety_agent.analyze_image(image_path, conf_threshold=confidence_threshold)
            else:
                raise HTTPException(status_code=400, detail="Provide an uploaded image file or image_path query parameter.")

            return {
                "timestamp": report["timestamp"],
                "worker_count": report["worker_count"],
                "detections": report["detections"],
                "worker_results": report["worker_results"],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/safety/alerts")
    def get_alerts(
        status: Optional[str] = Query(None, description="Filter by status: New, Acknowledged, Resolved"),
        limit: int = Query(100, ge=1, le=500),
    ):
        """Retrieve generated safety alerts."""
        alerts = db_manager.get_alerts(status=status, limit=limit)
        return {"count": len(alerts), "alerts": alerts}

    @app.patch("/api/safety/alerts/{alert_id}")
    def update_alert(alert_id: str, payload: AlertStatusUpdate):
        """Update safety alert status."""
        if payload.status not in ALERT_STATUSES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status '{payload.status}'. Must be one of {ALERT_STATUSES}"
            )
        success = db_manager.update_alert_status(alert_id, payload.status)
        if not success:
            raise HTTPException(status_code=404, detail=f"Alert with ID '{alert_id}' not found.")
        return {"status": "success", "alert_id": alert_id, "new_status": payload.status}

    @app.get("/api/safety/analytics")
    def get_analytics():
        """Retrieve safety analytics metrics and chart data."""
        return db_manager.get_analytics_summary()

    @app.post("/api/risk/projects/assess")
    def assess_project(payload: ProjectAssessmentRequest):
        """Run compliance and insurance agents for one construction project."""
        return risk_engine.assess_project(
            project_id=payload.project_id,
            inspection_reports=payload.inspection_reports,
            safety_logs=payload.safety_logs,
            incidents=payload.incidents,
            safety_trends=payload.safety_trends,
        )

    @app.get("/api/risk/projects/{project_id}/summary")
    def get_project_summary(project_id: str):
        """Return audit readiness, open violations, insurance risk, and alerts."""
        return db_manager.get_project_risk_summary(project_id)

    @app.get("/api/risk/projects/{project_id}/report")
    def get_project_report(project_id: str):
        """Return the persisted compliance and insurance intelligence report."""
        return db_manager.get_project_compliance_report(project_id)

    @app.get("/api/safety/workers")
    def get_workers(limit: int = Query(100, ge=1, le=500)):
        """Retrieve active worker monitoring records."""
        workers = db_manager.get_workers(limit=limit)
        return {"count": len(workers), "workers": workers}

    @app.get("/api/safety/sms-logs")
    def get_sms_logs(limit: int = Query(50, ge=1, le=200)):
        """Retrieve dispatched emergency SMS logs."""
        logs = sms_service.get_recent_logs(limit=limit)
        return {"count": len(logs), "logs": logs}

    @app.post("/api/safety/sms/send")
    def send_emergency_sms(payload: SendSMSRequest):
        """Manually trigger or test dispatch an emergency SMS."""
        if payload.risk_level.upper() not in ["HIGH", "CRITICAL"]:
            raise HTTPException(status_code=400, detail="SMS emergency alerts are restricted to HIGH and CRITICAL risks.")
        result = sms_service.send_direct_sms(
            worker_id=payload.worker_id,
            phone_number=payload.phone_number,
            risk_level=payload.risk_level,
            message=payload.message,
        )
        return {"status": "success", "dispatch": result}

    # ------------------------------------------------------------------
    # Milestone 4 — Enterprise Dashboard & Multi-Project Aggregation
    # ------------------------------------------------------------------
    @app.get("/api/v1/enterprise/dashboard")
    def get_enterprise_dashboard(project_id: Optional[str] = None):
        """Consolidated enterprise dashboard metrics, agent health, reports, and risk status."""
        return db_manager.get_enterprise_dashboard_data(project_id=project_id)

    @app.get("/api/v1/enterprise/projects")
    def get_enterprise_projects():
        """Retrieve portfolio risk rankings, portfolio average, and hotspot projects."""
        return intelligence_engine.get_portfolio_overview()

    @app.post("/api/v1/enterprise/projects")
    def register_or_update_project(payload: ProjectCreateRequest):
        """Register or update an enterprise project profile."""
        saved_id = db_manager.save_project({
            "project_id": payload.project_id,
            "project_name": payload.project_name or payload.project_id,
            "status": payload.status or "Active",
            "metadata": payload.metadata or {},
        })
        return {"status": "success", "project_id": saved_id}

    # ------------------------------------------------------------------
    # Milestone 4 — Global Risk Scoring & Predictive Forecasting
    # ------------------------------------------------------------------
    @app.get("/api/v1/risk/global-score")
    def get_global_risk_score(project_id: Optional[str] = None):
        """Calculates unified 0–100 Project Risk Score with weighted sub-agent indicators."""
        return intelligence_engine.calculate_global_risk_score(project_id=project_id)

    @app.post("/api/v1/risk/override")
    def override_risk_score(payload: RiskOverrideRequest):
        """Allows authorized supervisor to override risk score with mandatory audit justification."""
        result = intelligence_engine.apply_manual_risk_override(
            project_id=payload.project_id,
            new_risk_score=payload.new_risk_score,
            reason=payload.reason,
            user_name=payload.user_name or "Enterprise Supervisor",
        )
        return {"status": "success", "override": result}

    @app.get("/api/v1/risk/predictive-forecast")
    def get_predictive_forecast(project_id: Optional[str] = None):
        """Forecast incident probability %, recurring violation patterns, and safety ROI cost avoidance."""
        return intelligence_engine.run_predictive_analytics(project_id=project_id)

    # ------------------------------------------------------------------
    # Milestone 4 — Reporting Intelligence & Multi-Format Exports
    # ------------------------------------------------------------------
    @app.post("/api/v1/reports/generate")
    def generate_report(payload: ReportGenerateRequest):
        """Generate and persist an automated report (daily_operational, executive_summary, audit_report, project_health)."""
        rtype = payload.report_type.lower()
        fmt = (payload.export_format or "json").lower()

        if rtype == "daily_operational":
            result = reporting_agent.generate_daily_operational_report(
                project_id=payload.project_id, export_format=fmt
            )
        elif rtype == "executive_summary":
            result = reporting_agent.generate_executive_summary(
                project_id=payload.project_id, export_format=fmt
            )
        elif rtype == "audit_report":
            result = reporting_agent.generate_audit_packet(
                project_id=payload.project_id or "PRJ-METRO-01", export_format=fmt
            )
        elif rtype == "project_health":
            result = reporting_agent.generate_project_health_report(
                project_id=payload.project_id or "PRJ-METRO-01", export_format=fmt
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid report_type '{payload.report_type}'. Choose from: daily_operational, executive_summary, audit_report, project_health."
            )

        if fmt == "html":
            return HTMLResponse(content=result.get("rendered", str(result)))
        elif fmt in ("markdown", "md"):
            return PlainTextResponse(content=result.get("rendered", str(result)))
        return result

    @app.get("/api/v1/reports")
    def list_reports(
        project_id: Optional[str] = None,
        report_type: Optional[str] = None,
        limit: int = Query(50, ge=1, le=200),
    ):
        """Query persisted enterprise reports."""
        reports = db_manager.get_reports(project_id=project_id, report_type=report_type, limit=limit)
        return {"count": len(reports), "reports": reports}

    @app.get("/api/v1/reports/{report_id}")
    def get_report_by_id(report_id: str, format: Optional[str] = Query("json")):
        """Fetch a specific report by ID in JSON, Markdown, or HTML/PDF-ready format."""
        report = db_manager.get_report(report_id)
        if not report:
            raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found.")
        fmt = (format or "json").lower()
        if fmt == "json":
            return report
        rendered = reporting_agent.export_report_by_id(report_id, export_format=fmt)
        if fmt == "html":
            return HTMLResponse(content=rendered or "")
        return PlainTextResponse(content=rendered or "")

    # ------------------------------------------------------------------
    # Milestone 4 — Agent Performance & Immutable Audit Trail
    # ------------------------------------------------------------------
    @app.get("/api/v1/system/agent-health")
    def get_system_agent_health():
        """Retrieve real-time agent latency, efficiency, accuracy against enterprise baselines."""
        return intelligence_engine.get_agent_health_report()

    @app.get("/api/v1/audit/trail")
    def get_audit_trail(
        agent: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = Query(100, ge=1, le=500),
    ):
        """Retrieve immutable audit trail entries with optional agent and action filters."""
        trail = db_manager.get_audit_trail(agent=agent, action=action, limit=limit)
        return {"count": len(trail), "audit_trail": trail}


def _sanitize_report(report: Dict) -> Dict:
    """Remove numpy array fields before JSON serialization."""
    clean = dict(report)
    clean.pop("original_image", None)
    clean.pop("annotated_image", None)
    return clean


# Direct Python helper functions for programmatic access
def api_analyze_image(image_path: str, conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> Dict:
    report = safety_agent.analyze_image(image_path, conf_threshold)
    return _sanitize_report(report)

def api_get_alerts(status: Optional[str] = None, limit: int = 100) -> List[Dict]:
    return db_manager.get_alerts(status=status, limit=limit)

def api_update_alert_status(alert_id: str, new_status: str) -> bool:
    return db_manager.update_alert_status(alert_id, new_status)

def api_get_analytics() -> Dict:
    return db_manager.get_analytics_summary()

def api_get_workers(limit: int = 100) -> List[Dict]:
    return db_manager.get_workers(limit=limit)

def api_get_sms_logs(limit: int = 50) -> List[Dict]:
    return sms_service.get_recent_logs(limit=limit)

def api_send_emergency_sms(worker_id: str, phone_number: str, risk_level: str, message: str) -> Dict:
    return sms_service.send_direct_sms(worker_id, phone_number, risk_level, message)


if __name__ == "__main__":
    if HAS_FASTAPI:
        print("Starting Safety Intelligence REST API Server on http://0.0.0.0:8000...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("FastAPI not installed. Run 'pip install fastapi uvicorn' to run REST server.")
