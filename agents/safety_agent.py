"""
Safety Agent
============
The primary safety-analysis component for Milestone 2: Safety Intelligence & Worker Protection.

Pipeline:
    Image / Video Frame
          ↓
    YOLO Object Detection
          ↓
    Worker & PPE Association (PPEComplianceEngine)
          ↓
    Risk Assessment (RiskScorer)
          ↓
    Alert Generation (AlertSystem)
          ↓
    SQLite Persistence (SafetyDBManager)
          ↓
    Structured Safety Results Output
"""

import os
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.detector import ConstructionDetector
from risk.risk_scoring import RiskScorer
from agents.ppe_compliance_engine import PPEComplianceEngine
from agents.alert_system import AlertSystem
from database.db_manager import SafetyDBManager
from utils.config import (
    DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_REQUIRED_PPE,
    ALERT_COOLDOWN_SECONDS, RISK_LEVELS,
)
from utils.logger import AnalysisLogger


class SafetyAgent:
    """
    Main safety-analysis component that coordinates YOLO detection, PPE compliance,
    risk calculation, safety alerts, and database persistence.
    """

    def __init__(
        self,
        detector: Optional[ConstructionDetector] = None,
        db_manager: Optional[SafetyDBManager] = None,
        required_ppe: Optional[set] = None,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self.detector = detector or ConstructionDetector()
        self.db = db_manager or SafetyDBManager()
        available_classes = set(self.detector.class_names.values())
        configured_ppe = set(required_ppe) if required_ppe is not None else set(DEFAULT_REQUIRED_PPE)
        supported_required_ppe = configured_ppe.intersection(available_classes)
        self.ppe_engine = PPEComplianceEngine(
            required_ppe=supported_required_ppe,
            available_classes=available_classes,
        )
        self.scorer = RiskScorer()
        self.alert_system = AlertSystem(db_manager=self.db)
        self.logger = AnalysisLogger()
        self.conf_threshold = conf_threshold

    # ------------------------------------------------------------------
    # Image Analysis
    # ------------------------------------------------------------------
    def analyze_image(
        self,
        image_path: str,
        conf_threshold: Optional[float] = None,
    ) -> Dict:
        """Analyze an image file path and return structured safety results."""
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        detections, original_image = self.detector.detect_image(image_path, conf)
        return self._process_detections(
            detections, original_image, source=image_path, filename=os.path.basename(image_path), analysis_type="image"
        )

    def analyze_uploaded_image(
        self,
        image_bytes: bytes,
        filename: str = "uploaded_image.jpg",
        conf_threshold: Optional[float] = None,
    ) -> Dict:
        """Analyze raw image bytes (e.g. from dashboard file uploader)."""
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        detections, original_image = self.detector.detect_uploaded_image(image_bytes, conf)
        return self._process_detections(
            detections, original_image, source="upload", filename=filename, analysis_type="image"
        )

    # ------------------------------------------------------------------
    # Video Frame Analysis
    # ------------------------------------------------------------------
    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        filename: str = "video_frame.jpg",
        conf_threshold: Optional[float] = None,
    ) -> Dict:
        """Analyze a single BGR video frame."""
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        detections = self.detector.detect_frame(frame, conf)
        return self._process_detections(
            detections, frame, source=f"{filename}#f{frame_number}", filename=filename, analysis_type="video_frame"
        )

    # ------------------------------------------------------------------
    # Video Analysis Pipeline
    # ------------------------------------------------------------------
    def analyze_video(
        self,
        video_path: str,
        conf_threshold: Optional[float] = None,
        frame_skip: int = 10,
        max_frames: int = 300,
        progress_callback=None,
    ) -> Dict:
        """Process a full video file frame-by-frame."""
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_reports: List[Dict] = []
        frame_idx = 0
        analysed = 0

        while cap.isOpened() and analysed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                report = self.analyze_frame(
                    frame,
                    frame_number=frame_idx,
                    filename=os.path.basename(video_path),
                    conf_threshold=conf_threshold,
                )
                frame_reports.append(report)
                analysed += 1

                if progress_callback:
                    expected = max(1, min(max_frames, (total_frames + frame_skip - 1) // frame_skip))
                    progress_callback(analysed, expected)

            frame_idx += 1

        cap.release()

        # Aggregate video statistics
        worker_counts = [r["worker_count"] for r in frame_reports]
        compliant_counts = [r["compliant_workers"] for r in frame_reports]
        violation_counts = [r["violations"] for r in frame_reports]
        risk_scores = [r["risk"]["score"] for r in frame_reports]

        avg_workers = int(sum(worker_counts) / max(1, len(worker_counts)))
        avg_compliant = int(sum(compliant_counts) / max(1, len(compliant_counts)))
        total_violations = sum(violation_counts)
        max_score = max(risk_scores) if risk_scores else 0
        avg_score = int(sum(risk_scores) / max(1, len(risk_scores)))

        level_info = self.scorer._get_level(max(avg_score, max_score))

        return {
            "timestamp": datetime.now().isoformat(),
            "filename": os.path.basename(video_path),
            "analysis_type": "video",
            "total_frames": total_frames,
            "analysed_frames": len(frame_reports),
            "fps": fps,
            "worker_count": avg_workers,
            "compliant_workers": avg_compliant,
            "violations": total_violations,
            "risk_level": level_info["label"],
            "risk_score": max_score,
            "frame_reports": frame_reports,
            "all_alerts": [a for r in frame_reports for a in r.get("alerts", [])],
        }

    # ------------------------------------------------------------------
    # Core Pipeline Implementation
    # ------------------------------------------------------------------
    def _process_detections(
        self,
        detections: List[Dict],
        original_image: np.ndarray,
        source: str,
        filename: str,
        analysis_type: str = "image",
    ) -> Dict:
        now_iso = datetime.now().isoformat()

        # 1. PPE Compliance & Spatial Worker Association
        ppe_summary = self.ppe_engine.evaluate_compliance(detections)
        worker_results = ppe_summary["worker_results"]

        # 2. Risk Score & Level
        risk_report = self.scorer.calculate_risk(detections)
        risk_report = self._merge_worker_risk(risk_report, worker_results)

        # 3. Alert Generation with Cooldown
        alerts = self.alert_system.process_worker_violations(worker_results)

        # Determine primary violation description
        if ppe_summary["violations_count"] > 0:
            primary_violation = (
                worker_results[0]["violation"]
                if worker_results else "Worker detected without required PPE"
            )
        else:
            primary_violation = "None"

        # 4. Annotate Image with Bounding Boxes & Association Visuals
        annotated_image = self._annotate_safety_image(original_image, detections, worker_results)

        ppe_analytics = ppe_summary.get("ppe_compliance", {})

        # 5. Save to Database (SQLite Persistence)
        event_id = self.db.log_safety_event(
            timestamp=now_iso,
            source=source,
            worker_count=ppe_summary["worker_count"],
            compliant_workers=ppe_summary["compliant_workers"],
            violations_count=ppe_summary["violations_count"],
            risk_level=risk_report["level"],
            violation_type=primary_violation,
            raw_json={"worker_results": worker_results, "risk": risk_report},
        )

        for w in worker_results:
            is_viol = (w["ppe_status"] != "Compliant")
            if is_viol:
                self.db.log_ppe_violation(
                    event_id=event_id,
                    timestamp=now_iso,
                    worker_id=w["worker_id"],
                    violation_type=w["violation"],
                    missing_ppe=w["missing_ppe"],
                    risk_level=w["risk_level"],
                    confidence=w["confidence"],
                )
            
            self.db.update_worker_record(
                worker_id=w["worker_id"],
                timestamp=now_iso,
                ppe_status=w["ppe_status"],
                missing_ppe=w["missing_ppe"],
                risk_level=w["risk_level"],
                is_violation=is_viol,
            )

        # Log for JSON history compatibility
        self.logger.log_analysis(
            filename=filename,
            total_objects=len(detections),
            hazard_count=ppe_summary["violations_count"],
            risk_score=risk_report["score"],
            risk_level=risk_report["level"],
            worker_count=ppe_summary["worker_count"],
            analysis_type=analysis_type,
        )

        return {
            "timestamp": now_iso,
            "filename": filename,
            "source": source,
            "analysis_type": analysis_type,
            "worker_count": ppe_summary["worker_count"],
            "compliant_workers": ppe_summary["compliant_workers"],
            "violations": ppe_summary["violations_count"],
            "risk_level": risk_report["level"],
            "violation_type": primary_violation,
            "overall_safety_score": round(
                (ppe_summary["compliant_workers"] / max(1, ppe_summary["worker_count"])) * 100.0, 1
            ),
            "worker_results": worker_results,
            "alerts": alerts,
            "risk": risk_report,
            "detections": detections,
            "ppe_compliance": ppe_analytics,
            "overall_ppe_compliance": ppe_analytics.get("overall_ppe_compliance", 0.0),
            "original_image": original_image,
            "annotated_image": annotated_image,
        }

    def _merge_worker_risk(self, risk_report: Dict, worker_results: List[Dict]) -> Dict:
        """Ensure the site risk reflects explainable worker PPE violations."""
        level_scores = {level["label"]: level["min"] for level in RISK_LEVELS}
        worker_minimum = max(
            (level_scores.get(worker["risk_level"], 0) for worker in worker_results),
            default=0,
        )
        if worker_minimum <= risk_report["score"]:
            return risk_report

        promoted = dict(risk_report)
        promoted["score"] = worker_minimum
        level = self.scorer._get_level(worker_minimum)
        promoted["level"] = level["label"]
        promoted["level_color"] = level["color"]
        promoted["level_emoji"] = level["emoji"]
        promoted["risk_explanation"] = (
            f"{level['label']} Risk — worker PPE compliance rules identified "
            "a missing required PPE item."
        )
        return promoted

    # ------------------------------------------------------------------
    # Visualization & Annotation
    # ------------------------------------------------------------------
    def _annotate_safety_image(
        self,
        image: np.ndarray,
        detections: List[Dict],
        worker_results: List[Dict],
    ) -> np.ndarray:
        annotated = self.detector.annotate_image(image, detections)

        # Draw worker identification badges & status callouts
        for w in worker_results:
            bbox = [int(c) for c in w["bbox"]]
            x1, y1, x2, y2 = bbox
            worker_id = w["worker_id"]
            status = w["ppe_status"]
            missing = w["missing_ppe"]

            # Color scheme: Green for compliant, Red for non-compliant
            status_color = (0, 220, 0) if status == "Compliant" else (0, 0, 235)
            badge_text = f"{worker_id}: {status}"
            if missing:
                badge_text += f" (No {', '.join(missing)})"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), status_color, 2)
            
            # Badge text background box at bottom of worker box
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, y2), (x1 + tw + 6, y2 + th + 8), status_color, -1)
            cv2.putText(
                annotated,
                badge_text,
                (x1 + 3, y2 + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated
