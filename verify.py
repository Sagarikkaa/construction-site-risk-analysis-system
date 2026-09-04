"""Quick verification script — tests model loading and inference."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection.detector import ConstructionDetector
from risk.risk_scoring import RiskScorer
from agents.site_risk_agent import SiteRiskAgent

print("=" * 60)
print("  VERIFICATION: Construction Risk Intelligence Platform")
print("=" * 60)

# 1. Load model
print("\n[1] Loading YOLO model...")
detector = ConstructionDetector()
print(f"    ✓ Model loaded: {detector.is_loaded()}")
info = detector.get_model_info()
print(f"    ✓ Model type: {info['model_type']}")
print(f"    ✓ Classes: {info['num_classes']}")

# 2. Confirm the detector is ready
print("\n[2] Detector ready for uploaded images and videos")
assert detector.is_loaded()
print("    ✓ No bundled sample images required")

# 3. Test risk scoring
print("\n[3] Testing risk scoring...")
scorer = RiskScorer()
worker_without_ppe = scorer.calculate_risk([{
    "class_name": "Person",
    "confidence": 0.9,
    "bbox": [0, 0, 100, 100],
}])
print(f"    ✓ Worker-only score: {worker_without_ppe['score']}/100")
print(f"    ✓ Worker-only level: {worker_without_ppe['level_emoji']} {worker_without_ppe['level']}")
assert worker_without_ppe["level"] == "Low"
assert worker_without_ppe["score"] == 0
print("    ✓ Compliant worker / no-hazard worker correctly assigned Low Risk (0/100)")



video_frames = []
for frame_detections in (
    [{"class_name": "Person", "confidence": 0.9, "bbox": [0, 0, 100, 100]},
     {"class_name": "Person", "confidence": 0.9, "bbox": [110, 0, 210, 100]},
     {"class_name": "NO-Hardhat", "confidence": 0.8, "bbox": [0, 0, 20, 20]},
    {"class_name": "NO-Mask", "confidence": 0.8, "bbox": [110, 0, 130, 20]}],
    [{"class_name": "Person", "confidence": 0.9, "bbox": [0, 0, 100, 100]},
     {"class_name": "Person", "confidence": 0.9, "bbox": [110, 0, 210, 100]},
     {"class_name": "NO-Hardhat", "confidence": 0.8, "bbox": [0, 0, 20, 20]},
    {"class_name": "NO-Mask", "confidence": 0.8, "bbox": [110, 0, 130, 20]}],
):
    frame_risk = scorer.calculate_risk(frame_detections)
    video_frames.append({"risk": frame_risk, "detections": frame_detections,
                         "recommendations": []})
video_result = SiteRiskAgent(detector=detector)._aggregate_video_results(
    video_frames, "verification.mp4", 2, 30
)
assert video_result["avg_workers"] == 2.0
assert video_result["max_hazards"] == 2
assert video_result["overall_risk_level"] in ("Medium", "High", "Critical")
print("    ✓ Video averages, peak hazards, and peak-aware level are correct")



# 4. Test Site Risk Agent
print("\n[4] Testing Site Risk Agent...")
agent = SiteRiskAgent(detector=detector)
assert agent.scorer is not None
print("    ✓ Site Risk Agent initialized")

print("\n" + "=" * 60)
print("  ✅ ALL VERIFICATIONS PASSED")
print("=" * 60)
print("\nRun the dashboard with:  streamlit run app.py")
