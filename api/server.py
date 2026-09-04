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
from database.db_manager import SafetyDBManager
from utils.config import DEFAULT_CONFIDENCE_THRESHOLD, ALERT_STATUSES

try:
    from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Body
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


# Shared instances
safety_agent = SafetyAgent()
db_manager = SafetyDBManager()


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

    @app.get("/")
    def root():
        return {
            "platform": "Construction Risk Intelligence Platform",
            "milestone": "Milestone 2: Safety Intelligence & Worker Protection",
            "status": "Online",
            "endpoints": [
                "/api/safety/analyze",
                "/api/safety/detect",
                "/api/safety/alerts",
                "/api/safety/analytics",
                "/api/safety/workers",
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

    @app.get("/api/safety/workers")
    def get_workers(limit: int = Query(100, ge=1, le=500)):
        """Retrieve active worker monitoring records."""
        workers = db_manager.get_workers(limit=limit)
        return {"count": len(workers), "workers": workers}


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


if __name__ == "__main__":
    if HAS_FASTAPI:
        print("Starting Safety Intelligence REST API Server on http://0.0.0.0:8000...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("FastAPI not installed. Run 'pip install fastapi uvicorn' to run REST server.")
