"""
PPE Compliance Detection & Spatial Association Engine
======================================================
Associates YOLO object detections (workers & PPE items) using bounding-box
spatial relationships and evaluates individual worker PPE compliance.
"""

import math
import cv2
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from utils.config import ASSOCIATION_SCORE_THRESHOLD
try:
    from utils.config import FAIL_SAFE_PPE_ENFORCEMENT
except ImportError:
    FAIL_SAFE_PPE_ENFORCEMENT = True


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
        fail_safe: Optional[bool] = None,
    ):
        """
        Parameters
        ----------
        required_ppe : set of str, optional
            Set of required PPE item names (e.g. {"Hardhat", "Safety Vest"}).
        distance_threshold_ratio : float
            Multiplier relative to worker box size to associate gear with worker.
        available_classes : set of str, optional
            Set of classes supported by the model.
        fail_safe : bool, optional
            If True, workers detected without required PPE in their body region
            are treated as Non-Compliant rather than Uncertain. Defaults to
            FAIL_SAFE_PPE_ENFORCEMENT from config.
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
        self.fail_safe = fail_safe if fail_safe is not None else FAIL_SAFE_PPE_ENFORCEMENT

    def evaluate_compliance(self, detections: List[Dict], image: Optional[np.ndarray] = None) -> Dict:
        """
        Process detections and return worker-to-PPE association and compliance results.
        Includes spatial disambiguation for overlapping workers and color-based verification
        for safety vests and masks.

        Returns
        -------
        dict containing:
            worker_count: int
            compliant_workers: int
            violations_count: int
            worker_results: list of worker compliance dicts
            sanitized_detections: list of detections with suppressed false hazards
        """
        if not detections:
            return {
                "worker_count": 0,
                "compliant_workers": 0,
                "violations_count": 0,
                "worker_results": [],
                "sanitized_detections": [],
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

        # Multi-worker separation: If multiple hardhats exist inside a single worker box, separate workers
        hardhat_boxes = [d for d in gear_items if d["class_name"] in {"Hardhat", "NO-Hardhat"}]
        refined_workers = []
        for w in workers:
            w_box = w["bbox"]
            w_w = max(1.0, w_box[2] - w_box[0])
            contained_hats = [
                h for h in hardhat_boxes
                if (w_box[0] - 0.10 * w_w) <= (h["bbox"][0] + h["bbox"][2]) / 2 <= (w_box[2] + 0.10 * w_w)
                and (w_box[1] - 0.25 * (w_box[3] - w_box[1])) <= (h["bbox"][1] + h["bbox"][3]) / 2 <= (w_box[1] + 0.50 * (w_box[3] - w_box[1]))
            ]
            if len(contained_hats) >= 2:
                contained_hats.sort(key=lambda h: (h["bbox"][0] + h["bbox"][2]) / 2)
                h1_cx = (contained_hats[0]["bbox"][0] + contained_hats[0]["bbox"][2]) / 2
                h2_cx = (contained_hats[-1]["bbox"][0] + contained_hats[-1]["bbox"][2]) / 2
                if (h2_cx - h1_cx) >= 0.15 * w_w:
                    mid_x = (h1_cx + h2_cx) / 2
                    w1 = dict(w, bbox=[w_box[0], w_box[1], mid_x, w_box[3]])
                    w2 = dict(w, bbox=[mid_x, w_box[1], w_box[2], w_box[3]])
                    refined_workers.extend([w1, w2])
                    continue
            refined_workers.append(w)
        workers = refined_workers

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

            # Intelligent verification guards for hardhat, safety vest and mask
            if hardhat.get("positive") is None and self._has_hardhat_presence(image, w_bbox):
                w_h = max(1.0, w_bbox[3] - w_bbox[1])
                hat_box = [w_bbox[0], w_bbox[1], w_bbox[2], w_bbox[1] + 0.25 * w_h]
                hardhat = {
                    "positive": {
                        "class_name": "Hardhat",
                        "confidence": 0.90,
                        "bbox": hat_box,
                        "association_score": 0.95,
                    },
                    "negative": None,
                }
                gear_for_worker["Hardhat"] = hardhat["positive"]
                gear_for_worker.pop("NO-Hardhat", None)

            if self._has_high_vis_vest(image, w_bbox):
                w_h = max(1.0, w_bbox[3] - w_bbox[1])
                vest_box = [w_bbox[0], w_bbox[1] + 0.20 * w_h, w_bbox[2], w_bbox[1] + 0.75 * w_h]
                vest = {
                    "positive": {
                        "class_name": "Safety Vest",
                        "confidence": 0.95,
                        "bbox": vest_box,
                        "association_score": 0.98,
                    },
                    "negative": None,
                }
                gear_for_worker["Safety Vest"] = vest["positive"]
                gear_for_worker.pop("NO-Safety Vest", None)

            if mask.get("negative") is not None:
                neg_conf = mask["negative"].get("confidence", 0.0)
                if neg_conf < 0.65 or self._has_protective_mask(image, w_bbox):
                    mask["negative"] = None
                    gear_for_worker.pop("NO-Mask", None)

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

            # 1. Check Hardhat
            if "Hardhat" in self.required_ppe:
                if has_hardhat and not has_no_hardhat:
                    detected_ppe.append("Hardhat")
                elif has_no_hardhat or self.fail_safe:
                    missing_ppe.append("helmet")
                else:
                    uncertain_ppe.append("helmet")

            # 2. Check Safety Vest
            if "Safety Vest" in self.required_ppe:
                if has_vest and not has_no_vest:
                    detected_ppe.append("Safety Vest")
                elif has_no_vest or self.fail_safe:
                    missing_ppe.append("vest")
                else:
                    uncertain_ppe.append("vest")

            # 3. Check Mask (if in required_ppe)
            if "Mask" in self.required_ppe:
                if has_mask and not has_no_mask:
                    detected_ppe.append("Mask")
                elif has_no_mask or self.fail_safe:
                    missing_ppe.append("mask")
                else:
                    uncertain_ppe.append("mask")
            elif has_mask and not has_no_mask:
                # Masks may be optional for the site policy, but a positive
                # detection must still contribute to PPE analytics.
                detected_ppe.append("Mask")

            # 4. Optional compliance categories when the model provides support.
            if "Safety Boots" in self.required_ppe or "Boots" in self.required_ppe:
                if has_boots and not has_no_boots:
                    detected_ppe.append("Safety Boots")
                elif has_no_boots or self.fail_safe:
                    missing_ppe.append("boots")
                else:
                    uncertain_ppe.append("boots")
            elif has_boots and not has_no_boots:
                detected_ppe.append("Safety Boots")
            elif has_no_boots:
                missing_ppe.append("boots")

            if "Protective Gloves" in self.required_ppe or "Gloves" in self.required_ppe:
                if has_gloves and not has_no_gloves:
                    detected_ppe.append("Protective Gloves")
                elif has_no_gloves or self.fail_safe:
                    missing_ppe.append("gloves")
                else:
                    uncertain_ppe.append("gloves")
            elif has_gloves and not has_no_gloves:
                detected_ppe.append("Protective Gloves")
            elif has_no_gloves:
                missing_ppe.append("gloves")

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

        suppressed_hazards = set()
        for worker in worker_results:
            if "Safety Vest" in worker.get("detected_ppe", []):
                suppressed_hazards.add("NO-Safety Vest")
            if "mask" not in worker.get("missing_ppe", []):
                suppressed_hazards.add("NO-Mask")

        sanitized_detections = [
            d for d in detections
            if not (d["class_name"] == "NO-Safety Vest" and "NO-Safety Vest" in suppressed_hazards)
            and not (d["class_name"] == "NO-Mask" and d.get("confidence", 1.0) < 0.65)
        ]

        result = {
            "worker_count": len(worker_results),
            "compliant_workers": compliant_count,
            "violations_count": violation_count,
            "worker_results": worker_results,
            "sanitized_detections": sanitized_detections,
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

    @staticmethod
    def _has_high_vis_vest(image: Optional[np.ndarray], worker_bbox: List[float]) -> bool:
        """Verify presence of fluorescent safety vest in worker's torso region."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return False
        try:
            h_img, w_img = image.shape[:2]
            x1 = max(0, int(worker_bbox[0]))
            y1 = max(0, int(worker_bbox[1] + 0.15 * (worker_bbox[3] - worker_bbox[1])))
            x2 = min(w_img, int(worker_bbox[2]))
            y2 = min(h_img, int(worker_bbox[1] + 0.80 * (worker_bbox[3] - worker_bbox[1])))
            if x2 <= x1 or y2 <= y1:
                return False
            torso = image[y1:y2, x1:x2]
            if torso.size == 0 or torso.shape[0] < 10 or torso.shape[1] < 10:
                return False
            hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
            # Fluorescent orange: H [4, 25], S >= 80, V >= 75
            orange_mask = cv2.inRange(hsv, np.array([4, 80, 75]), np.array([25, 255, 255]))
            # Fluorescent lime-yellow: H [25, 45], S >= 70, V >= 75
            yellow_mask = cv2.inRange(hsv, np.array([25, 70, 75]), np.array([45, 255, 255]))
            high_vis = cv2.bitwise_or(orange_mask, yellow_mask)
            ratio = cv2.countNonZero(high_vis) / float(torso.shape[0] * torso.shape[1])
            return ratio >= 0.08
        except Exception:
            return False

    @staticmethod
    def _has_protective_mask(image: Optional[np.ndarray], worker_bbox: List[float]) -> bool:
        """Check if worker's lower face region contains a protective mask (e.g. white N95 / surgical)."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return False
        try:
            h_img, w_img = image.shape[:2]
            x1 = max(0, int(worker_bbox[0] + 0.15 * (worker_bbox[2] - worker_bbox[0])))
            y1 = max(0, int(worker_bbox[1] + 0.08 * (worker_bbox[3] - worker_bbox[1])))
            x2 = min(w_img, int(worker_bbox[2] - 0.15 * (worker_bbox[2] - worker_bbox[0])))
            y2 = min(h_img, int(worker_bbox[1] + 0.35 * (worker_bbox[3] - worker_bbox[1])))
            if x2 <= x1 or y2 <= y1:
                return False
            face = image[y1:y2, x1:x2]
            if face.size == 0 or face.shape[0] < 5 or face.shape[1] < 5:
                return False
            hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
            # White mask / respirator: Low saturation, high brightness
            white_mask = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([180, 60, 255]))
            # Blue surgical mask: H [90, 125], S >= 60, V >= 60
            blue_mask = cv2.inRange(hsv, np.array([90, 60, 60]), np.array([125, 255, 255]))
            mask_area = cv2.bitwise_or(white_mask, blue_mask)
            ratio = cv2.countNonZero(mask_area) / float(face.shape[0] * face.shape[1])
            return ratio >= 0.10
        except Exception:
            return False

    @staticmethod
    def _has_hardhat_presence(image: Optional[np.ndarray], worker_bbox: List[float]) -> bool:
        """Check if worker's head region contains a hardhat (white, yellow, blue, orange, red)."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return False
        try:
            h_img, w_img = image.shape[:2]
            w_w = worker_bbox[2] - worker_bbox[0]
            w_h = worker_bbox[3] - worker_bbox[1]
            x1 = max(0, int(worker_bbox[0] + 0.05 * w_w))
            y1 = max(0, int(worker_bbox[1] - 0.15 * w_h))
            x2 = min(w_img, int(worker_bbox[2] - 0.05 * w_w))
            y2 = min(h_img, int(worker_bbox[1] + 0.35 * w_h))
            if x2 <= x1 or y2 <= y1:
                return False
            head = image[y1:y2, x1:x2]
            if head.size == 0 or head.shape[0] < 5 or head.shape[1] < 5:
                return False
            hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
            # White hardhat: V >= 170, S <= 60
            white_mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 60, 255]))
            # Yellow hardhat: H [15, 38], S >= 75, V >= 85
            yellow_mask = cv2.inRange(hsv, np.array([15, 75, 85]), np.array([38, 255, 255]))
            # Blue hardhat: H [95, 130], S >= 70, V >= 70
            blue_mask = cv2.inRange(hsv, np.array([95, 70, 70]), np.array([130, 255, 255]))
            # Orange/Red hardhat: H [0, 12], S >= 80, V >= 80
            orange_mask = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([12, 255, 255]))
            combined = cv2.bitwise_or(white_mask, cv2.bitwise_or(yellow_mask, cv2.bitwise_or(blue_mask, orange_mask)))
            ratio = cv2.countNonZero(combined) / float(head.shape[0] * head.shape[1])
            return ratio >= 0.08
        except Exception:
            return False


