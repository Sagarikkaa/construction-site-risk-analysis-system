"""
Construction Site Object Detector
==================================
Wraps YOLOv8 inference for construction-site images and video frames.
Provides structured detection results used by downstream risk analysis.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import sys
import ultralytics
import ultralytics.engine, ultralytics.nn, ultralytics.utils

# Register legacy Ultralytics unpickling alias so PyTorch loads older .pt weights cleanly
sys.modules["ultralytics.yolo"] = ultralytics
sys.modules["ultralytics.yolo.utils"] = ultralytics.utils
sys.modules["ultralytics.yolo.engine"] = ultralytics.engine
sys.modules["ultralytics.yolo.nn"] = ultralytics.nn

from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    MODEL_PATH, CLASS_NAMES, DEFAULT_CONFIDENCE_THRESHOLD,
    HAZARD_CLASSES, SAFETY_EQUIPMENT_CLASSES, WORKER_CLASSES, HEAVY_OBJECTS,
    HAZARD_BOX_COLOR, SAFETY_BOX_COLOR, WORKER_BOX_COLOR,
    HEAVY_BOX_COLOR, DEFAULT_BOX_COLOR, BOX_THICKNESS, FONT_SCALE,
)


class ConstructionDetector:
    """Loads a YOLOv8 model and runs inference on construction-site imagery."""

    def __init__(self, model_path: str = MODEL_PATH, verbose: bool = False):
        """
        Parameters
        ----------
        model_path : str
            Path to the YOLO ``best.pt`` weights file.
        verbose : bool
            If True, YOLO prints per-inference logs.
        """
        self.model_path = model_path
        self.verbose = verbose
        self.model: Optional[YOLO] = None
        self.class_names: Dict[int, str] = {}
        self.hazard_classes = set()
        self.safety_equipment_classes = set()
        self.worker_classes = set()
        self.heavy_object_classes = set()
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Load the YOLO model from disk. Raises FileNotFoundError if missing."""
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"YOLO model not found at '{self.model_path}'. "
                f"Please place 'best.pt' inside the models/ directory."
            )
        self.model = YOLO(self.model_path)
        self.class_names = {
            int(class_id): str(class_name)
            for class_id, class_name in self.model.names.items()
        }
        available = set(self.class_names.values())
        self.hazard_classes = available.intersection(HAZARD_CLASSES)
        self.safety_equipment_classes = available.intersection(SAFETY_EQUIPMENT_CLASSES)
        self.worker_classes = available.intersection(WORKER_CLASSES)
        self.heavy_object_classes = available.intersection(HEAVY_OBJECTS)

    def is_loaded(self) -> bool:
        return self.model is not None

    def get_model_info(self) -> Dict:
        """Return basic metadata about the loaded model."""
        if not self.is_loaded():
            return {"status": "Not loaded", "error": "Model file missing"}
        return {
            "status": "Loaded",
            "model_path": self.model_path,
            "model_type": "YOLOv8n (nano)",
            "classes": list(self.class_names.values()),
            "num_classes": len(self.class_names),
            "hazard_classes": sorted(self.hazard_classes),
            "safety_equipment_classes": sorted(self.safety_equipment_classes),
            "worker_classes": sorted(self.worker_classes),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def detect_image(
        self,
        image_path: str,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Run detection on a single image file.

        Returns
        -------
        detections : list[dict]
            Each dict: {class_name, class_id, confidence, bbox: [x1,y1,x2,y2]}
        image : np.ndarray
            The original image (BGR, as read by OpenCV).
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not decode image: {image_path}")

        return self._run_inference(image, conf_threshold), image

    def detect_frame(
        self,
        frame: np.ndarray,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> List[Dict]:
        """Run detection on a single video frame (BGR numpy array)."""
        if frame is None or frame.size == 0:
            return []
        return self._run_inference(frame, conf_threshold)

    def detect_uploaded_image(
        self,
        image_bytes: bytes,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> Tuple[List[Dict], np.ndarray]:
        """Run detection on raw image bytes (e.g. from Streamlit upload)."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode the uploaded image bytes.")
        return self._run_inference(image, conf_threshold), image

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _run_inference(
        self, image: np.ndarray, conf_threshold: float
    ) -> List[Dict]:
        """Core inference logic shared by all public detect_* methods."""
        results = self.model(image, conf=conf_threshold, verbose=self.verbose)
        detections: List[Dict] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "class_name": cls_name,
                    "class_id": cls_id,
                    "confidence": float(box.conf[0]),
                    "bbox": [x1, y1, x2, y2],
                })

        return self._filter_false_hazards(detections, image)


    def _filter_false_hazards(self, detections: List[Dict], image: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Non-Regressive Detection & Filter Pipeline:
        1. Conflict Resolution: Suppress false negative hazard boxes (NO-Hardhat, NO-Mask) if positive gear is present.
        2. HSV High-Vis Color Verification: Convert NO-Safety Vest to Safety Vest if high-vis orange/yellow pixels exist.
        3. Worker PPE Verification: Ensures unequipped workers in excavation pits get flagged while protected workers are preserved.
        """
        if not detections:
            return detections

        pos_hardhats = [d for d in detections if d["class_name"] == "Hardhat"]
        pos_vests = [d for d in detections if d["class_name"] == "Safety Vest"]
        pos_masks = [d for d in detections if d["class_name"] == "Mask"]

        hsv = None
        if image is not None and image.size > 0:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_orange, upper_orange = np.array([5, 90, 90]), np.array([25, 255, 255])
            lower_yellow, upper_yellow = np.array([20, 90, 90]), np.array([35, 255, 255])

        cleaned = []
        for d in detections:
            name = d["class_name"]
            box = [int(c) for c in d["bbox"]]

            if name == "NO-Mask" and pos_masks:
                continue
            if name == "NO-Hardhat" and pos_hardhats:
                continue
            if name == "NO-Safety Vest" and hsv is not None:
                x1, y1 = max(0, box[0]), max(0, box[1])
                x2, y2 = min(image.shape[1], box[2]), min(image.shape[0], box[3])
                if x2 > x1 and y2 > y1:
                    roi = hsv[y1:y2, x1:x2]
                    if roi.size > 0:
                        m_o = cv2.inRange(roi, lower_orange, upper_orange)
                        m_y = cv2.inRange(roi, lower_yellow, upper_yellow)
                        if (np.sum(cv2.bitwise_or(m_o, m_y) > 0) / (roi.shape[0] * roi.shape[1])) > 0.03:
                            cleaned.append({
                                "class_name": "Safety Vest",
                                "class_id": 7,
                                "confidence": d["confidence"],
                                "bbox": d["bbox"],
                            })
                            pos_vests.append(d)
                            continue

            cleaned.append(d)

        workers = [d for d in detections if d["class_name"] in self.worker_classes]
        for w in workers:
            wx1, wy1, wx2, wy2 = [int(c) for c in w["bbox"]]
            w_h = wy2 - wy1
            head_box = [wx1, wy1, wx2, wy1 + int(0.35 * w_h)]
            has_hh = any(self._is_near_worker(hh["bbox"], w["bbox"]) for hh in pos_hardhats)
            has_mask = any(self._is_near_worker(m["bbox"], w["bbox"]) for m in pos_masks)
            has_vest = any(self._is_near_worker(v["bbox"], w["bbox"]) for v in pos_vests)

            if not has_vest and hsv is not None:
                roi = hsv[max(0, wy1):min(image.shape[0], wy2), max(0, wx1):min(image.shape[1], wx2)]
                if roi.size > 0:
                    m_o = cv2.inRange(roi, lower_orange, upper_orange)
                    m_y = cv2.inRange(roi, lower_yellow, upper_yellow)
                    if (np.sum(cv2.bitwise_or(m_o, m_y) > 0) / (roi.shape[0] * roi.shape[1])) > 0.04:
                        has_vest = True

            has_no_hh_already = any(d["class_name"] == "NO-Hardhat" and self._is_near_worker(d["bbox"], w["bbox"]) for d in cleaned)

            if not has_hh and not (has_mask and has_vest) and not has_no_hh_already:
                cleaned.append({
                    "class_name": "NO-Hardhat",
                    "class_id": 2,
                    "confidence": round(w["confidence"], 2),
                    "bbox": head_box,
                })

        return cleaned

    @staticmethod
    def _is_near_worker(gear_box: List[float], worker_box: List[float]) -> bool:
        gc = ((gear_box[0] + gear_box[2]) / 2, (gear_box[1] + gear_box[3]) / 2)
        wc = ((worker_box[0] + worker_box[2]) / 2, (worker_box[1] + worker_box[3]) / 2)
        w_size = max(worker_box[2] - worker_box[0], worker_box[3] - worker_box[1])
        return np.hypot(gc[0] - wc[0], gc[1] - wc[1]) <= 1.2 * w_size




    @staticmethod
    def _boxes_overlap(boxA: List[float], boxB: List[float]) -> bool:
        x_left = max(boxA[0], boxB[0])
        y_top = max(boxA[1], boxB[1])
        x_right = min(boxA[2], boxB[2])
        y_bottom = min(boxA[3], boxB[3])

        if x_right <= x_left or y_bottom <= y_top:
            return False

        intersection = (x_right - x_left) * (y_bottom - y_top)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        min_area = min(areaA, areaB)

        return min_area > 0 and (intersection / min_area) > 0.20




    # ------------------------------------------------------------------
    # Annotation helper
    # ------------------------------------------------------------------
    def annotate_image(
        self,
        image: np.ndarray,
        detections: List[Dict],
    ) -> np.ndarray:
        """
        Draw colour-coded bounding boxes and labels on a copy of the image.

        Colours
        -------
        Red    – hazard classes (NO-Hardhat, NO-Mask, NO-Safety Vest)
        Green  – safety equipment (Hardhat, Mask, Safety Vest, Safety Cone)
        Orange – workers (Person)
        Dark orange – heavy objects (machinery, vehicle)
        Grey   – anything else
        """
        annotated = image.copy()

        for det in detections:
            name = det["class_name"]
            conf = det["confidence"]
            x1, y1, x2, y2 = [int(c) for c in det["bbox"]]

            # Pick colour by semantic group
            if name in self.hazard_classes:
                color = HAZARD_BOX_COLOR
            elif name in self.safety_equipment_classes:
                color = SAFETY_BOX_COLOR
            elif name in self.worker_classes:
                color = WORKER_BOX_COLOR
            elif name in self.heavy_object_classes:
                color = HEAVY_BOX_COLOR
            else:
                color = DEFAULT_BOX_COLOR

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)

            label = f"{name} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                annotated, label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

        return annotated
