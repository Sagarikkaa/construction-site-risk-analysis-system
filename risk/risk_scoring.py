"""
Risk Scoring Engine
====================
Computes a 0–100 site risk score from YOLO detection results using
configured hazard weights, bounding-box proximity analysis, and a
multi-hazard multiplier.

NOTE: The scoring methodology is a **prototype** designed for academic
demonstration. It is NOT a scientifically validated safety standard.
"""

import math
from typing import List, Dict, Tuple
from collections import Counter

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    HAZARD_CLASSES, WORKER_CLASSES, HAZARD_WEIGHTS,
    MAX_RAW_SCORE, RISK_LEVELS, PROXIMITY_DISTANCE_RATIO,
)


def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Return (cx, cy) of an [x1, y1, x2, y2] bounding box."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _bbox_diagonal(bbox: List[float]) -> float:
    """Return the diagonal length of a bounding box."""
    return math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])


def _center_distance(a: List[float], b: List[float]) -> float:
    """Euclidean distance between centres of two bounding boxes."""
    ca, cb = _bbox_center(a), _bbox_center(b)
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1])


class RiskScorer:
    """Calculates a normalised site risk score from detection results."""

    def __init__(self):
        self.hazard_weights = HAZARD_WEIGHTS
        self.max_raw = MAX_RAW_SCORE
        self.proximity_ratio = PROXIMITY_DISTANCE_RATIO

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def calculate_risk(self, detections: List[Dict]) -> Dict:
        """
        Analyse a list of YOLO detections and return a risk report.

        Parameters
        ----------
        detections : list[dict]
            Each dict must have keys: class_name, confidence, bbox.

        Returns
        -------
        dict with keys:
            score          – int 0-100
            level          – str (Low / Medium / High / Critical)
            level_color    – hex colour string
            level_emoji    – emoji for the level
            hazard_breakdown – list of per-hazard detail dicts
            contributing_factors – list[str]
            risk_factors – evidence-based factors and model limitations
            risk_explanation – short evidence-based explanation
            worker_count   – int
            hazard_count   – int
            total_objects  – int
        """
        if not detections:
            return self._empty_report()

        # Partition detections
        hazards = [d for d in detections if d["class_name"] in HAZARD_CLASSES]
        workers = [d for d in detections if d["class_name"] in WORKER_CLASSES]
        detected_objects = [d["class_name"] for d in detections]
        raw_score = 0.0
        hazard_breakdown: List[Dict] = []
        contributing_factors: List[str] = []
        ppe_hazard_type_set: set = set()

        # --- Accurate Worker Count Calculation ---
        hardhats = [d for d in detections if d["class_name"] in {"Hardhat", "NO-Hardhat"}]
        vests = [d for d in detections if d["class_name"] in {"Safety Vest", "NO-Safety Vest"}]
        actual_worker_count = max(len(workers), len(hardhats), len(vests))


        risk_factors = [
            f"{actual_worker_count} worker(s) detected" if actual_worker_count > 0 else "No workers detected",
            "Working-at-height hazard: NOT SUPPORTED BY CURRENT MODEL",
        ]

        # --- 1. PPE-related hazards ---
        hazard_counts = Counter(d["class_name"] for d in hazards)
        for haz_name, count in hazard_counts.items():
            weight = self.hazard_weights.get(haz_name, 10)
            confidence_sum = sum(
                d["confidence"] for d in hazards if d["class_name"] == haz_name
            )
            contribution = weight * min(2, count) * min(1.0, confidence_sum / count)
            raw_score += contribution
            ppe_hazard_type_set.add(haz_name)

            avg_conf = sum(
                d["confidence"] for d in hazards if d["class_name"] == haz_name
            ) / count

            hazard_breakdown.append({
                "hazard": haz_name,
                "count": count,
                "avg_confidence": round(avg_conf, 2),
                "risk_contribution": contribution,
                "severity": self._severity_label(weight),
            })
            contributing_factors.append(
                f"{count}× {haz_name} detected (weight {weight} each)"
            )

        # Heavy machinery & vehicle proximity penalties (mitigated by PPE compliance)
        heavy_objs = [d for d in detections if d["class_name"] in {"machinery", "vehicle"}]
        ppe_mitigation_factor = 0.25 if len(hazards) == 0 else 1.0
        for heavy in heavy_objs:
            nearby_w = self._count_nearby_workers(heavy, workers)
            if nearby_w > 0:
                prox_weight = int((20 * nearby_w) * ppe_mitigation_factor)
                raw_score += prox_weight
                if ppe_mitigation_factor < 1.0:
                    contributing_factors.append(
                        f"{heavy['class_name'].title()} near {nearby_w} worker(s) — PPE safety gear mitigates proximity risk (+{prox_weight} risk)"
                    )
                else:
                    contributing_factors.append(
                        f"{heavy['class_name'].title()} operating near {nearby_w} worker(s) (+{prox_weight} proximity risk)"
                    )


        if ppe_hazard_type_set:
            risk_factors.extend(f"{name} detected" for name in sorted(ppe_hazard_type_set))
        affected_worker_indices = self._workers_affected_by_hazards(hazards, workers)
        if affected_worker_indices:
            risk_factors.append(
                f"{len(affected_worker_indices)} worker(s) associated with detected PPE violation(s)"
            )
            if len(affected_worker_indices) > 1:
                contribution = 5 * (len(affected_worker_indices) - 1)
                raw_score += contribution
                contributing_factors.append(
                    f"Multiple affected workers → +{contribution} additional risk"
                )

        if len(ppe_hazard_type_set) >= 2:
            raw_score += 10
            contributing_factors.append(
                "Multiple PPE violation types → +10 additional risk"
            )

        for worker_hazards in affected_worker_indices.values():
            if len(worker_hazards) >= 2:
                raw_score += 10
                contributing_factors.append(
                    "Multiple PPE violations associated with one worker → +10 additional risk"
                )
                break


        # --- 2. Normalise to 0-100 ---
        score = int(min(100, (raw_score / self.max_raw) * 100))
        level_info = self._get_level(score)

        if ppe_hazard_type_set:
            risk_explanation = (
                f"{level_info['label']} Risk — PPE violations detected. "
                "The current model does not support direct working-at-height hazard detection."
            )
        elif workers:
            risk_explanation = (
                f"{level_info['label']} Risk — workers detected, but no model-supported PPE "
                "violation was detected. Working-at-height hazards are not supported."
            )
        else:
            risk_explanation = (
                f"{level_info['label']} Risk — no model-supported worker or PPE hazard detected. "
                "Working-at-height hazards are not supported."
            )

        return {
            "score": score,
            "level": level_info["label"],
            "level_color": level_info["color"],
            "level_emoji": level_info["emoji"],
            "hazard_breakdown": hazard_breakdown,
            "contributing_factors": contributing_factors,
            "worker_count": len(workers),
            "hazard_count": len(hazards),
            "total_objects": len(detections),
            "detected_objects": sorted(set(detected_objects)),
            "ppe_violations": {
                name: sum(1 for d in hazards if d["class_name"] == name)
                for name in sorted(ppe_hazard_type_set)
            },
            "raw_score": round(raw_score, 1),
            "risk_factors": risk_factors,
            "risk_explanation": risk_explanation,
        }

    # ------------------------------------------------------------------
    # Proximity analysis
    # ------------------------------------------------------------------
    def _count_nearby_workers(
        self, heavy_det: Dict, workers: List[Dict]
    ) -> int:
        """Count workers whose bounding-box centre is 'near' a heavy object."""
        if not workers:
            return 0

        h_bbox = heavy_det["bbox"]
        h_diag = _bbox_diagonal(h_bbox)
        count = 0

        for w in workers:
            w_bbox = w["bbox"]
            w_diag = _bbox_diagonal(w_bbox)
            avg_diag = (h_diag + w_diag) / 2
            dist = _center_distance(h_bbox, w_bbox)
            if dist <= self.proximity_ratio * avg_diag:
                count += 1

        return count

    def _workers_affected_by_hazards(
        self, hazards: List[Dict], workers: List[Dict]
    ) -> Dict[int, set]:
        """Associate PPE detections with nearby worker boxes for explainable bonuses."""
        affected: Dict[int, set] = {}
        for hazard in hazards:
            for worker_index, worker in enumerate(workers):
                h_diag = _bbox_diagonal(hazard["bbox"])
                w_diag = _bbox_diagonal(worker["bbox"])
                if _center_distance(hazard["bbox"], worker["bbox"]) <= self.proximity_ratio * ((h_diag + w_diag) / 2):
                    affected.setdefault(worker_index, set()).add(hazard["class_name"])
        return affected

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _severity_label(weight: int) -> str:
        if weight >= 50:
            return "High"
        return "Moderate"

    @staticmethod
    def _get_level(score: int) -> Dict:
        for lvl in RISK_LEVELS:
            if lvl["min"] <= score <= lvl["max"]:
                return lvl
        return RISK_LEVELS[-1]  # fallback to Critical

    @staticmethod
    def _empty_report() -> Dict:
        return {
            "score": 0,
            "level": "Low",
            "level_color": "#22c55e",
            "level_emoji": "🟢",
            "hazard_breakdown": [],
            "contributing_factors": ["No objects detected in the frame."],
            "worker_count": 0,
            "hazard_count": 0,
            "total_objects": 0,
            "detected_objects": [],
            "ppe_violations": {},
            "raw_score": 0,
            "risk_factors": [
                "No workers detected",
                "Working-at-height hazard: NOT SUPPORTED BY CURRENT MODEL",
            ],
            "risk_explanation": (
                "Low Risk — no model-supported worker or PPE hazard detected. "
                "Working-at-height hazards are not supported."
            ),
        }
