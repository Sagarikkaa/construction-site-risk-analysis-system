"""
Site Risk Agent
================
The agentic layer of the Construction Risk Intelligence Platform.

This module orchestrates the full analysis pipeline:
    Image/Frame → YOLO Detection → Hazard Interpretation →
    Risk Scoring → Recommendations

It is designed as a **standalone, modular component** so that Milestone 2
can easily connect additional agents (Safety Agent, Compliance Agent, etc.)
through an inter-agent communication interface.
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.detector import ConstructionDetector
from risk.risk_scoring import RiskScorer
from utils.config import (
    RECOMMENDATIONS, NO_HAZARD_MESSAGE,
    HAZARD_CLASSES, HEAVY_OBJECTS, DEFAULT_CONFIDENCE_THRESHOLD,
)
from utils.logger import AnalysisLogger


class SiteRiskAgent:
    """
    Autonomous agent that monitors construction-site imagery for hazards.

    Responsibilities
    ----------------
    1. Receive / load construction-site images or video frames.
    2. Run YOLO object detection.
    3. Interpret detections to identify hazardous conditions.
    4. Calculate an overall site risk score via the RiskScorer.
    5. Generate human-readable safety recommendations.
    6. Log every analysis for historical tracking.

    The agent maintains an in-memory history of recent analyses and
    persists them to a JSON log file.
    """

    def __init__(self, detector: Optional[ConstructionDetector] = None):
        """
        Parameters
        ----------
        detector : ConstructionDetector, optional
            A pre-initialised detector. If None, one is created automatically.
        """
        self.detector = detector or ConstructionDetector()
        self.scorer = RiskScorer()
        self.logger = AnalysisLogger()
        self._history: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API — image analysis
    # ------------------------------------------------------------------
    def analyze_image(
        self,
        image_path: str,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> Dict:
        """
        Full pipeline: load image → detect → score → recommend → log.

        Returns a comprehensive analysis report dict.
        """
        detections, original_image = self.detector.detect_image(
            image_path, conf_threshold
        )
        annotated_image = self.detector.annotate_image(original_image, detections)

        risk_report = self.scorer.calculate_risk(detections)
        recommendations = self._generate_recommendations(risk_report)

        report = {
            "timestamp": datetime.now().isoformat(),
            "filename": os.path.basename(image_path),
            "source": image_path,
            "analysis_type": "image",
            "detections": detections,
            "original_image": original_image,
            "annotated_image": annotated_image,
            "risk": risk_report,
            "recommendations": recommendations,
        }

        # Persist
        self._history.append(report)
        self.logger.log_analysis(
            filename=report["filename"],
            total_objects=risk_report["total_objects"],
            hazard_count=risk_report["hazard_count"],
            risk_score=risk_report["score"],
            risk_level=risk_report["level"],
            worker_count=risk_report["worker_count"],
            analysis_type="image",
        )

        return report

    def analyze_uploaded_image(
        self,
        image_bytes: bytes,
        filename: str = "uploaded_image",
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> Dict:
        """Analyse raw image bytes (from Streamlit uploader)."""
        detections, original_image = self.detector.detect_uploaded_image(
            image_bytes, conf_threshold
        )
        annotated_image = self.detector.annotate_image(original_image, detections)

        risk_report = self.scorer.calculate_risk(detections)
        recommendations = self._generate_recommendations(risk_report)

        report = {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "source": "upload",
            "analysis_type": "image",
            "detections": detections,
            "original_image": original_image,
            "annotated_image": annotated_image,
            "risk": risk_report,
            "recommendations": recommendations,
        }

        self._history.append(report)
        self.logger.log_analysis(
            filename=filename,
            total_objects=risk_report["total_objects"],
            hazard_count=risk_report["hazard_count"],
            risk_score=risk_report["score"],
            risk_level=risk_report["level"],
            worker_count=risk_report["worker_count"],
            analysis_type="image",
        )

        return report

    # ------------------------------------------------------------------
    # Public API — video analysis
    # ------------------------------------------------------------------
    def analyze_video_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        filename: str = "video_frame",
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> Dict:
        """Analyse a single video frame. Returns the same report structure."""
        detections = self.detector.detect_frame(frame, conf_threshold)
        annotated_frame = self.detector.annotate_image(frame, detections)
        risk_report = self.scorer.calculate_risk(detections)
        recommendations = self._generate_recommendations(risk_report)

        return {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "frame_number": frame_number,
            "analysis_type": "video_frame",
            "detections": detections,
            "original_image": frame,
            "annotated_image": annotated_frame,
            "risk": risk_report,
            "recommendations": recommendations,
        }

    def analyze_video(
        self,
        video_path: str,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        frame_skip: int = 10,
        max_frames: int = 300,

        progress_callback=None,
    ) -> Dict:
        """
        Process a video file frame-by-frame.

        Parameters
        ----------
        video_path : str
            Path to the video file.
        conf_threshold : float
            YOLO confidence threshold.
        frame_skip : int
            Analyse every Nth frame to limit processing time.
        max_frames : int
            Maximum number of frames to analyse.
        progress_callback : callable, optional
            Called with (current_frame, total_frames) for progress bars.

        Returns
        -------
        dict with aggregated video-level statistics and per-frame reports.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_reports: List[Dict] = []
        frame_idx = 0
        analysed = 0
        debug_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "outputs", "video_debug",
        )
        os.makedirs(debug_dir, exist_ok=True)

        while cap.isOpened() and analysed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                report = self.analyze_video_frame(
                    frame, frame_idx,
                    filename=os.path.basename(video_path),
                    conf_threshold=conf_threshold,
                )
                frame_reports.append(report)
                detections = report["detections"]
                print(
                    f"Frame {frame_idx}: "
                    + (", ".join(
                        f"{d['class_name']} ({d['confidence']:.2f}) "
                        f"[{', '.join(f'{value:.0f}' for value in d['bbox'])}]"
                        for d in detections
                    ) or "no detections")
                )
                if analysed < 3:
                    debug_path = os.path.join(debug_dir, f"frame_{frame_idx}.jpg")
                    cv2.imwrite(debug_path, report["annotated_image"])
                analysed += 1

                if progress_callback:
                    expected_total = max(1, min(max_frames, (total_frames + frame_skip - 1) // frame_skip)) if total_frames > 0 else max_frames
                    progress_callback(analysed, expected_total)


            frame_idx += 1

        cap.release()

        # Aggregate statistics
        aggregated = self._aggregate_video_results(
            frame_reports, os.path.basename(video_path), total_frames, fps
        )

        # Log the overall video analysis
        self.logger.log_analysis(
            filename=os.path.basename(video_path),
            total_objects=aggregated["avg_objects"],
            hazard_count=aggregated["avg_hazards"],
            risk_score=aggregated["avg_risk_score"],
            risk_level=aggregated["overall_risk_level"],
            worker_count=aggregated["avg_workers"],
            analysis_type="video",
        )

        return aggregated

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------
    def _generate_recommendations(self, risk_report: Dict) -> List[str]:
        """Produce context-aware recommendations from the risk report."""
        if not risk_report["hazard_breakdown"]:
            return [NO_HAZARD_MESSAGE]

        recs: List[str] = []
        seen_keys = set()

        for haz in risk_report["hazard_breakdown"]:
            haz_name = haz["hazard"]
            # Direct PPE hazard
            if haz_name in RECOMMENDATIONS and haz_name not in seen_keys:
                recs.append(RECOMMENDATIONS[haz_name])
                seen_keys.add(haz_name)
            # Proximity hazard (e.g. "machinery near workers")
            for key in ("machinery_proximity", "vehicle_proximity"):
                if key.split("_")[0] in haz_name and key not in seen_keys:
                    recs.append(RECOMMENDATIONS[key])
                    seen_keys.add(key)

        # Multi-hazard advisory
        if len(risk_report["hazard_breakdown"]) >= 3 and "multiple_hazards" not in seen_keys:
            recs.append(RECOMMENDATIONS["multiple_hazards"])

        # High-risk advisory
        if risk_report["score"] >= 50 and "high_risk" not in seen_keys:
            recs.append(RECOMMENDATIONS["high_risk"])

        return recs if recs else [NO_HAZARD_MESSAGE]

    # ------------------------------------------------------------------
    # Video aggregation
    # ------------------------------------------------------------------
    def _aggregate_video_results(
        self,
        frame_reports: List[Dict],
        filename: str,
        total_frames: int,
        fps: float,
    ) -> Dict:
        """Combine per-frame reports into overall video statistics."""
        if not frame_reports:
            return {
                "filename": filename,
                "total_frames": total_frames,
                "analysed_frames": 0,
                "fps": fps,
                "avg_risk_score": 0,
                "max_risk_score": 0,
                "overall_risk_level": "Low",
                "avg_objects": 0,
                "avg_hazards": 0,
                "max_hazards": 0,
                "avg_workers": 0,
                "max_workers": 0,
                "peak_aware_score": 0,
                "detected_classes": [],
                "detected_objects": [],
                "ppe_violations": {},
                "risk_explanation": "Low Risk — no model-supported detections were found. Working-at-height hazards are not supported.",
                "frame_reports": [],
                "recommendations": [NO_HAZARD_MESSAGE],
            }

        scores = [r["risk"]["score"] for r in frame_reports]
        objects = [r["risk"]["total_objects"] for r in frame_reports]
        hazards = [r["risk"]["hazard_count"] for r in frame_reports]
        workers = [r["risk"]["worker_count"] for r in frame_reports]
        detected_classes = sorted({
            detection["class_name"]
            for report in frame_reports
            for detection in report["detections"]
        })
        ppe_violations: Dict[str, int] = {}
        for report in frame_reports:
            for name, count in report["risk"].get("ppe_violations", {}).items():
                ppe_violations[name] = ppe_violations.get(name, 0) + count

        avg_score = int(sum(scores) / len(scores))
        max_score = max(scores)

        # Use the highest-risk frame's report for recommendations
        worst = max(frame_reports, key=lambda r: r["risk"]["score"])

        # Video safety risk is driven by Peak Hazard Exposure so severe hazard moments
        # in a video (e.g. unequipped workers / heavy equipment proximity) are never hidden
        # or averaged down by low background frames.
        if max_score >= 50:
            peak_aware_score = max_score
        else:
            peak_aware_score = max(avg_score, max_score)

        level_info = self.scorer._get_level(peak_aware_score)

        if any(
            report["risk"].get("hazard_breakdown")
            for report in frame_reports
        ):
            overall_explanation = (
                f"{level_info['label']} Risk — supported PPE violation evidence was detected. "
                "The current model does not support direct working-at-height hazard detection."
            )
        else:
            overall_explanation = (
                f"{level_info['label']} Risk — no model-supported hazard evidence was detected. "
                "The current model does not support direct working-at-height hazard detection."
            )

        return {
            "filename": filename,
            "total_frames": total_frames,
            "analysed_frames": len(frame_reports),
            "fps": fps,
            "avg_risk_score": avg_score,
            "max_risk_score": max_score,
            "overall_risk_level": level_info["label"],
            "overall_risk_color": level_info["color"],
            "overall_risk_emoji": level_info["emoji"],
            "avg_objects": round(sum(objects) / len(objects), 2),
            "avg_hazards": round(sum(hazards) / len(hazards), 2),
            "max_hazards": max(hazards),
            "avg_workers": round(sum(workers) / len(workers), 2),
            "max_workers": max(workers),
            "peak_aware_score": peak_aware_score,
            "detected_classes": detected_classes,
            "detected_objects": detected_classes,
            "ppe_violations": ppe_violations,
            "risk_explanation": overall_explanation,
            "frame_reports": frame_reports,
            "recommendations": worst["recommendations"],
        }

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------
    def get_analysis_history(self, limit: int = 20) -> List[Dict]:
        """Return recent logged analyses."""
        return self.logger.get_history(limit)
