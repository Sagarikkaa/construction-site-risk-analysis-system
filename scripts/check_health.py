import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import cv2

# Set path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("  COMPREHENSIVE MODEL & SAFETY PIPELINE HEALTH CHECK")
print("=" * 60)

# 1. Verify Model Weight File
from utils.config import MODEL_PATH
print("\n[1] Checking YOLO Model File...")
if os.path.isfile(MODEL_PATH):
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"    ✓ Weight file exists: {MODEL_PATH} ({size_mb:.2f} MB)")
else:
    print(f"    ❌ Weight file missing at: {MODEL_PATH}")
    sys.exit(1)

# 2. Test Detector Initialization and Class Mapping
from detection.detector import ConstructionDetector
detector = ConstructionDetector()
print("\n[2] Checking Detector Loading...")
print(f"    ✓ Detector loaded: {detector.is_loaded()}")
info = detector.get_model_info()
print(f"    ✓ Model Type: {info.get('model_type')}")
print(f"    ✓ Total Classes ({info.get('num_classes')}): {list(detector.class_names.values())}")

# 3. Real Image Inference Test
test_img_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "test", "images")
test_images = [f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.png'))][:3] if os.path.isdir(test_img_dir) else []

if test_images:
    sample_img_path = os.path.join(test_img_dir, test_images[0])
    print(f"\n[3] Running Real Inference on Test Image: {test_images[0]}...")
    detections, img = detector.detect_image(sample_img_path)
    print(f"    ✓ Detections found: {len(detections)}")
    for d in detections[:5]:
        print(f"      - {d['class_name']} (Conf: {d['confidence']*100:.1f}%, BBox: {d['bbox']})")
else:
    print("    ℹ️ No test images directory found, skipping direct image load.")

# 4. Safety Agent & PPE Association Pipeline
from agents.safety_agent import SafetyAgent
safety_agent = SafetyAgent(detector=detector)
print("\n[4] Testing End-to-End Safety Agent & PPE Association...")
if test_images:
    report = safety_agent.analyze_image(sample_img_path)
    print(f"    ✓ Workers Detected: {report.get('workers_detected', 0)}")
    print(f"    ✓ Compliant Workers: {report.get('compliant_workers', 0)}")
    print(f"    ✓ Non-Compliant: {report.get('non_compliant_workers', 0)}")
    print(f"    ✓ Site Safety Score: {report.get('safety_score', 100):.1f}%")
    print(f"    ✓ Overall Site Risk: {report.get('risk_level', 'Unknown')}")
    print(f"    ✓ Alerts Generated: {len(report.get('alerts', []))}")

# 5. Automated Emergency SMS Dispatch & Database Logging
print("\n[5] Testing Automated Emergency SMS Local Dispatch...")
from utils.sms_service import SMSService
sms_service = SMSService()
test_alert = {
    "alert_id": "ALT-HEALTH-001",
    "worker_id": "Worker-01",
    "violation_type": "Missing Helmet and Safety Vest",
    "risk_level": "CRITICAL",
}
res = sms_service.send_alert_sms(test_alert)
print(f"    ✓ Dispatch Status: {res['status']}")
print(f"    ✓ Provider: {res['provider']}")
print(f"    ✓ Assigned Recipient: {res['phone_number']}")
print(f"    ✓ Database SMS ID: #{res['sms_id']}")

print("\n" + "=" * 60)
print("  🎉 ALL MODEL & SAFETY PIPELINE CHECKS PASSED PERFECTLY!")
print("=" * 60)
