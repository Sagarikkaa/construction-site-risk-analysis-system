"""
PPE Compliance Detection & Spatial Association Engine
======================================================
Associates YOLO object detections (workers & PPE items) using bounding-box
spatial relationships and evaluates individual worker PPE compliance.
"""

import math
import numpy as np
from typing import List, Dict, Set, Tuple, Optional


class PPEComplianceEngine:
    """
    Associates workers (Person) with PPE equipment objects and evaluates safety compliance.
    """

    def __init__(
        self,
        required_ppe: Optional[Set[str]] = None,
        distance_threshold_ratio: float = 0.8,
        available_classes: Optional[Set[str]] = None,
    ):
        """
        Parameters
        ----------
        required_ppe : set of str, optional
            Set of required PPE item names (e.g. {"Hardhat", "Safety Vest"}).
        distance_threshold_ratio : float
            Multiplier relative to worker box size to associate gear with worker.
        """
        default_required = {"Hardhat", "Safety Vest"}
        if available_classes is not None:
            default_required &= set(available_classes)
        configured_required = set(required_ppe) if required_ppe is not None else default_required
        self.required_ppe = (
            configured_required.intersection(set(available_classes))
            if available_classes is not None
            else configured_required
        )
        self.distance_threshold_ratio = distance_threshold_ratio
        self.available_classes = set(available_classes) if available_classes is not None else None

    def evaluate_compliance(self, detections: List[Dict]) -> Dict:
        """
        Process detections and return worker-to-PPE association and compliance results.

        Returns
        -------
        dict containing:
            worker_count: int
            compliant_workers: int
            violations_count: int
            worker_results: list of worker compliance dicts
        """
        if not detections:
            return {
                "worker_count": 0,
                "compliant_workers": 0,
                "violations_count": 0,
                "worker_results": [],
            }

        worker_classes = {"Person"}
        gear_classes = {
            "Hardhat", "Safety Vest", "Mask",
            "NO-Hardhat", "NO-Safety Vest", "NO-Mask"
        }
        if self.available_classes is not None:
            worker_classes &= self.available_classes
            gear_classes &= self.available_classes

        workers = [d for d in detections if d["class_name"] in worker_classes]
        gear_items = [d for d in detections if d["class_name"] in gear_classes]

        worker_boxes = [w["bbox"] for w in workers]
        worker_confs = [w["confidence"] for w in workers]

        # Synthesize worker instances if no Person box was detected but PPE items exist
        if not workers and gear_items:
            for det in gear_items:
                box = det["bbox"]
                if not any(self._bbox_overlap(box, wb) for wb in worker_boxes):
                    w_box = [
                        max(0, box[0] - 25),
                        max(0, box[1] - 25),
                        box[2] + 25,
                        box[3] + 160,
                    ]
                    worker_boxes.append(w_box)
                    worker_confs.append(det["confidence"])

        # Associate each gear item to the single best/closest qualifying worker
        worker_gear = {
            i: {cls_name: [] for cls_name in gear_classes}
            for i in range(len(worker_boxes))
        }

        for gear in gear_items:
            best_idx = None
            min_dist = float("inf")

            for idx, w_bbox in enumerate(worker_boxes):
                if self._can_associate(gear["bbox"], w_bbox):
                    dist = self._center_distance(gear["bbox"], w_bbox)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = idx

            if best_idx is not None:
                worker_gear[best_idx][gear["class_name"]].append(gear)

        worker_results = []
        compliant_count = 0
        violation_count = 0

        for idx, w_bbox in enumerate(worker_boxes):
            worker_id = f"Worker-{idx + 1:02d}"
            confidence = round(worker_confs[idx] if idx < len(worker_confs) else 0.85, 2)

            gear_for_worker = worker_gear[idx]
            has_hardhat = bool(gear_for_worker.get("Hardhat", []))
            has_no_hardhat = bool(gear_for_worker.get("NO-Hardhat", []))
            has_vest = bool(gear_for_worker.get("Safety Vest", []))
            has_no_vest = bool(gear_for_worker.get("NO-Safety Vest", []))
            has_mask = bool(gear_for_worker.get("Mask", []))
            has_no_mask = bool(gear_for_worker.get("NO-Mask", []))

            detected_ppe = []
            missing_ppe = []

            # 1. Check Hardhat / Helmet
            if "Hardhat" in self.required_ppe:
                if has_hardhat and not has_no_hardhat:
                    detected_ppe.append("Hardhat")
                else:
                    missing_ppe.append("helmet")

            # 2. Check Safety Vest
            if "Safety Vest" in self.required_ppe:
                if has_vest and not has_no_vest:
                    detected_ppe.append("Safety Vest")
                else:
                    missing_ppe.append("vest")

            # 3. Check Mask (if in required_ppe)
            if "Mask" in self.required_ppe:
                if has_mask and not has_no_mask:
                    detected_ppe.append("Mask")
                else:
                    missing_ppe.append("mask")

            # Status & Risk Level Calculation
            if not missing_ppe:
                status = "Compliant"
                risk_level = "Low"
                violation_msg = "None"
                should_alert = False
                compliant_count += 1
            else:
                status = "Non-Compliant"
                violation_count += 1
                should_alert = True

                # Rule-based explainable risk assessment
                if "helmet" in missing_ppe and "vest" in missing_ppe:
                    risk_level = "Critical" if len(missing_ppe) > 2 else "High"
                    violation_msg = "Worker detected without required PPE (Missing Helmet and Vest)"
                elif "helmet" in missing_ppe:
                    risk_level = "High"
                    violation_msg = "Worker detected without required PPE (Missing Helmet)"
                elif "vest" in missing_ppe:
                    risk_level = "Medium"
                    violation_msg = "Worker detected without required PPE (Missing Safety Vest)"
                else:
                    risk_level = "Medium"
                    violation_msg = f"Worker detected without required PPE (Missing {', '.join(missing_ppe)})"

            worker_results.append({
                "worker_id": worker_id,
                "ppe_status": status,
                "missing_ppe": missing_ppe,
                "detected_ppe": detected_ppe,
                "risk_level": risk_level,
                "violation": violation_msg,
                "alert": should_alert,
                "confidence": confidence,
                "bbox": [round(c, 1) for c in w_bbox],
            })

        return {
            "worker_count": len(worker_results),
            "compliant_workers": compliant_count,
            "violations_count": violation_count,
            "worker_results": worker_results,
        }

    def _can_associate(self, gear_box: List[float], worker_box: List[float]) -> bool:
        """Check if gear box is geometrically compatible with worker box."""
        if self._bbox_overlap(gear_box, worker_box):
            return True

        gc_x = (gear_box[0] + gear_box[2]) / 2
        gc_y = (gear_box[1] + gear_box[3]) / 2
        w_w = max(1.0, worker_box[2] - worker_box[0])
        w_h = max(1.0, worker_box[3] - worker_box[1])

        # Allow horizontal tolerance (gear roughly within worker horizontal span)
        horiz_ok = (worker_box[0] - 0.35 * w_w) <= gc_x <= (worker_box[2] + 0.35 * w_w)
        # Allow vertical tolerance (hardhat can sit above worker, vest covers torso)
        vert_ok = (worker_box[1] - 0.40 * w_h) <= gc_y <= (worker_box[3] + 0.20 * w_h)

        if horiz_ok and vert_ok:
            return True

        # Fallback centroid distance
        dist = self._center_distance(gear_box, worker_box)
        w_diag = math.hypot(w_w, w_h)
        return dist <= (self.distance_threshold_ratio * w_diag)

    @staticmethod
    def _center_distance(boxA: List[float], boxB: List[float]) -> float:
        ca = ((boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2)
        cb = ((boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2)
        return math.hypot(ca[0] - cb[0], ca[1] - cb[1])

    @staticmethod
    def _bbox_overlap(boxA: List[float], boxB: List[float]) -> bool:
        """Check if boxA intersects boxB."""
        x_left = max(boxA[0], boxB[0])
        y_top = max(boxA[1], boxB[1])
        x_right = min(boxA[2], boxB[2])
        y_bottom = min(boxA[3], boxB[3])

        if x_right <= x_left or y_bottom <= y_top:
            return False

        intersection = (x_right - x_left) * (y_bottom - y_top)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        return (intersection / max(1.0, areaA)) > 0.12
