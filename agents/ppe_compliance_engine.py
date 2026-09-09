"""
PPE Compliance Detection & Spatial Association Engine
======================================================
Associates YOLO object detections (workers & PPE items) using bounding-box
spatial relationships and evaluates individual worker PPE compliance.
"""

import math
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from utils.config import ASSOCIATION_SCORE_THRESHOLD


class PPEComplianceEngine:
    """
    Associates workers (Person) with PPE equipment objects and evaluates safety compliance.
    """

    PPE_CATEGORY_MAP = {
        "hard_hat": {
            "label": "Hard Hats",
            "positive": "Hardhat",
            "negative": "NO-Hardhat",
            "field": "helmet",
        },
        "safety_vest": {
            "label": "Safety Vests",
            "positive": "Safety Vest",
            "negative": "NO-Safety Vest",
            "field": "vest",
        },
        "safety_boots": {
            "label": "Safety Boots",
            "positive": "Safety Boots",
            "negative": "NO-Safety Boots",
            "field": "boots",
            "aliases": [("Boots", "NO-Boots")],
        },
        "protective_gloves": {
            "label": "Protective Gloves",
            "positive": "Protective Gloves",
            "negative": "NO-Protective Gloves",
            "field": "gloves",
            "aliases": [("Gloves", "NO-Gloves")],
        },
        "mask": {
            "label": "Safety Masks",
            "positive": "Mask",
            "negative": "NO-Mask",
            "field": "mask",
        },
    }

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

        # Only Person detections create workers. PPE boxes can never create a
        # synthetic worker because that produces false worker counts.
        workers = [d for d in detections if d["class_name"] in worker_classes]
        gear_items = [d for d in detections if d["class_name"] in gear_classes]

        # Assign each gear box to one person using its semantic body region,
        # not just distance to the full person box.
        worker_gear = {i: {} for i in range(len(workers))}
        for gear in gear_items:
            candidates = []
            for idx, worker in enumerate(workers):
                score = self._association_score(gear, worker)
                if score >= ASSOCIATION_SCORE_THRESHOLD:
                    candidates.append((score, idx))
            if candidates:
                score, best_idx = max(candidates, key=lambda item: item[0])
                current = worker_gear[best_idx].get(gear["class_name"])
                candidate = dict(gear, association_score=round(score, 3))
                if current is None or candidate["association_score"] > current["association_score"]:
                    worker_gear[best_idx][gear["class_name"]] = candidate

        worker_results = []
        compliant_count = 0
        violation_count = 0

        for idx, worker in enumerate(workers):
            worker_id = f"Worker-{idx + 1:02d}"
            w_bbox = worker["bbox"]
            confidence = round(worker["confidence"], 2)

            gear_for_worker = worker_gear[idx]
            hardhat = self._best_status_detection(gear_for_worker, "Hardhat", "NO-Hardhat")
            vest = self._best_status_detection(gear_for_worker, "Safety Vest", "NO-Safety Vest")
            mask = self._best_status_detection(gear_for_worker, "Mask", "NO-Mask")

            boots = self._best_status_detection(gear_for_worker, "Safety Boots", "NO-Safety Boots")
            if boots["positive"] is None and boots["negative"] is None:
                boots = self._best_status_detection(gear_for_worker, "Boots", "NO-Boots")

            gloves = self._best_status_detection(gear_for_worker, "Protective Gloves", "NO-Protective Gloves")
            if gloves["positive"] is None and gloves["negative"] is None:
                gloves = self._best_status_detection(gear_for_worker, "Gloves", "NO-Gloves")

            has_hardhat = hardhat["positive"] is not None
            has_no_hardhat = hardhat["negative"] is not None
            has_vest = vest["positive"] is not None
            has_no_vest = vest["negative"] is not None
            has_mask = mask["positive"] is not None
            has_no_mask = mask["negative"] is not None
            has_boots = boots["positive"] is not None
            has_no_boots = boots["negative"] is not None
            has_gloves = gloves["positive"] is not None
            has_no_gloves = gloves["negative"] is not None

            detected_ppe = []
            missing_ppe = []

            uncertain_ppe = []

            # An explicit NO-* detection is a confirmed absence. If neither a
            # positive nor negative class is associated, the evidence is
            # uncertain rather than an automatic high-risk violation.
            if "Hardhat" in self.required_ppe:
                if has_hardhat and not has_no_hardhat:
                    detected_ppe.append("Hardhat")
                elif has_no_hardhat:
                    missing_ppe.append("helmet")
                else:
                    uncertain_ppe.append("helmet")

            # 2. Check Safety Vest
            if "Safety Vest" in self.required_ppe:
                if has_vest and not has_no_vest:
                    detected_ppe.append("Safety Vest")
                elif has_no_vest:
                    missing_ppe.append("vest")
                else:
                    uncertain_ppe.append("vest")

            # 3. Check Mask (if in required_ppe)
            if "Mask" in self.required_ppe:
                if has_mask and not has_no_mask:
                    detected_ppe.append("Mask")
                elif has_no_mask:
                    missing_ppe.append("mask")
                else:
                    uncertain_ppe.append("mask")
            elif has_mask and not has_no_mask:
                # Masks may be optional for the site policy, but a positive
                # detection must still contribute to PPE analytics.
                detected_ppe.append("Mask")

            # 4. Optional compliance categories when the model provides support.
            if has_boots and not has_no_boots:
                detected_ppe.append("Safety Boots")
            elif has_no_boots:
                missing_ppe.append("boots")
            elif any(name in gear_for_worker for name in ("Safety Boots", "NO-Safety Boots", "Boots", "NO-Boots")):
                uncertain_ppe.append("boots")

            if has_gloves and not has_no_gloves:
                detected_ppe.append("Protective Gloves")
            elif has_no_gloves:
                missing_ppe.append("gloves")
            elif any(name in gear_for_worker for name in ("Protective Gloves", "NO-Protective Gloves", "Gloves", "NO-Gloves")):
                uncertain_ppe.append("gloves")

            # Status & Risk Level Calculation
            if not missing_ppe and not uncertain_ppe:
                status = "Compliant"
                risk_level = "Low"
                violation_msg = "None"
                should_alert = False
                compliant_count += 1
            elif missing_ppe:
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
            else:
                status = "Detection Uncertain"
                risk_level = "Low"
                violation_msg = f"Detection uncertain for {', '.join(uncertain_ppe)}"
                should_alert = False

            worker_results.append({
                "worker_id": worker_id,
                "ppe_status": status,
                "missing_ppe": missing_ppe,
                "uncertain_ppe": uncertain_ppe,
                "detected_ppe": detected_ppe,
                "risk_level": risk_level,
                "violation": violation_msg,
                "alert": should_alert,
                "confidence": confidence,
                "bbox": [round(c, 1) for c in w_bbox],
                "person_confidence": confidence,
                "person_bbox": [round(c, 1) for c in w_bbox],
                "associated_helmet_confidence": self._confidence(hardhat["positive"]),
                "associated_helmet_bbox": self._bbox(hardhat["positive"]),
                "associated_vest_confidence": self._confidence(vest["positive"]),
                "associated_vest_bbox": self._bbox(vest["positive"]),
                "helmet_association_score": self._score(hardhat["positive"]),
                "vest_association_score": self._score(vest["positive"]),
                "association_score": round(max(
                    self._score(hardhat["positive"]), self._score(vest["positive"])
                ), 3),
            })

        result = {
            "worker_count": len(worker_results),
            "compliant_workers": compliant_count,
            "violations_count": violation_count,
            "worker_results": worker_results,
        }
        result["ppe_compliance"] = self.calculate_ppe_compliance(worker_results)
        return result

    def calculate_ppe_compliance(self, worker_results: List[Dict]) -> Dict:
        """Compute dynamic per-PPE compliance percentages from worker results."""
        total_workers = max(1, len(worker_results))
        compliance = {}
        enabled = []

        for key, config in self.PPE_CATEGORY_MAP.items():
            if key not in {"hard_hat", "safety_vest", "mask", "safety_boots", "protective_gloves"}:
                continue
            if self.available_classes is not None:
                supported_names = {
                    config["positive"],
                    config["negative"],
                    *(name for pair in config.get("aliases", []) for name in pair),
                }
                if not supported_names.intersection(self.available_classes):
                    continue

            compliant = 0
            for worker in worker_results:
                detected = set(worker.get("detected_ppe", []))
                missing = set(worker.get("missing_ppe", []))
                status = False

                if key == "hard_hat":
                    status = "Hardhat" in detected and "helmet" not in missing
                elif key == "safety_vest":
                    status = "Safety Vest" in detected and "vest" not in missing
                elif key == "mask":
                    status = "Mask" in detected and "mask" not in missing
                elif key == "safety_boots":
                    status = "Safety Boots" in detected and "boots" not in missing
                elif key == "protective_gloves":
                    status = "Protective Gloves" in detected and "gloves" not in missing

                if status:
                    compliant += 1

            percentage = round((compliant / total_workers) * 100, 1) if total_workers else 0.0
            compliance[key] = {
                "compliant": compliant,
                "total": total_workers,
                "percentage": percentage,
                "label": config["label"],
                "status": self._compliance_status(percentage),
                "enabled": True,
            }
            enabled.append(key)

        overall = (
            round(sum(item["percentage"] for item in compliance.values()) / len(enabled), 1)
            if enabled else 0.0
        )

        return {
            "total_workers": total_workers,
            "ppe_compliance": compliance,
            "overall_ppe_compliance": overall,
            "enabled_categories": enabled,
        }

    @staticmethod
    def _compliance_status(percentage: float) -> str:
        if percentage >= 90:
            return "High compliance"
        if percentage >= 75:
            return "Moderate compliance"
        return "Low compliance"

    def _association_score(self, gear: Dict, worker: Dict) -> float:
        """Score gear against the correct semantic region of one worker."""
        worker_box = worker["bbox"]
        x1, y1, x2, y2 = worker_box
        width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
        name = gear["class_name"]
        if name in {"Hardhat", "NO-Hardhat", "Mask", "NO-Mask"}:
            region = [x1 - 0.15 * width, y1 - 0.25 * height, x2 + 0.15 * width, y1 + 0.35 * height]
        else:
            region = [x1 - 0.15 * width, y1 + 0.20 * height, x2 + 0.15 * width, y1 + 0.78 * height]

        gear_box = gear["bbox"]
        center_x = (gear_box[0] + gear_box[2]) / 2
        center_y = (gear_box[1] + gear_box[3]) / 2
        region_width = max(1.0, region[2] - region[0])
        region_height = max(1.0, region[3] - region[1])
        horizontal = max(0.0, 1.0 - abs(center_x - (x1 + x2) / 2) / (region_width / 2))
        vertical = max(0.0, 1.0 - abs(center_y - (region[1] + region[3]) / 2) / (region_height / 2))
        region_score = (horizontal + vertical) / 2
        overlap_score = self._intersection_over_area(gear_box, region)
        center_score = 1.0 if self._point_in_box((center_x, center_y), region) else region_score
        return round(0.45 * overlap_score + 0.35 * center_score + 0.20 * region_score, 3)

    @staticmethod
    def _best_status_detection(gear: Dict, positive_name: str, negative_name: str) -> Dict:
        positive = gear.get(positive_name)
        negative = gear.get(negative_name)
        if positive and negative:
            positive_weight = positive["confidence"] * positive["association_score"]
            negative_weight = negative["confidence"] * negative["association_score"]
            return {"positive": positive, "negative": None} if positive_weight >= negative_weight else {"positive": None, "negative": negative}
        return {"positive": positive, "negative": negative}

    @staticmethod
    def _confidence(detection: Optional[Dict]) -> Optional[float]:
        return round(detection["confidence"], 2) if detection else None

    @staticmethod
    def _bbox(detection: Optional[Dict]) -> Optional[List[float]]:
        return [round(value, 1) for value in detection["bbox"]] if detection else None

    @staticmethod
    def _score(detection: Optional[Dict]) -> float:
        return round(detection["association_score"], 3) if detection else 0.0

    @staticmethod
    def _point_in_box(point: Tuple[float, float], box: List[float]) -> bool:
        return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]

    @staticmethod
    def _intersection_over_area(box_a: List[float], box_b: List[float]) -> float:
        left, top = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
        right, bottom = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        area = max(1.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        return intersection / area

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
