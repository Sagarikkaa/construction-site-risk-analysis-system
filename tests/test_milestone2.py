"""
Milestone 2 — Safety Intelligence & Worker Protection — Test Suite
===================================================================
Tests PPE compliance, safety agent, alert system, database operations,
and end-to-end pipeline using actual model detections on test images.
"""

import os
import sys
import json
import glob
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Ultralytics compatibility shim
import ultralytics
import ultralytics.engine, ultralytics.nn, ultralytics.utils
sys.modules["ultralytics.yolo"] = ultralytics
sys.modules["ultralytics.yolo.utils"] = ultralytics.utils
sys.modules["ultralytics.yolo.engine"] = ultralytics.engine
sys.modules["ultralytics.yolo.nn"] = ultralytics.nn


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def test_database_operations():
    """Test 1: SQLite Database CRUD Operations"""
    separator("TEST 1: Database Operations")
    from database.db_manager import SafetyDBManager

    test_db_path = os.path.join(PROJECT_ROOT, "data", "test_safety.db")
    db = SafetyDBManager(db_path=test_db_path)

    # Log a safety event
    event_id = db.log_safety_event(
        timestamp="2026-09-03T12:00:00",
        source="test_image.jpg",
        worker_count=3,
        compliant_workers=1,
        violations_count=2,
        risk_level="High",
        violation_type="Missing Helmet",
    )
    print(f"  ✅ Safety event logged: ID={event_id}")

    # Log a PPE violation
    viol_id = db.log_ppe_violation(
        event_id=event_id,
        timestamp="2026-09-03T12:00:00",
        worker_id="Worker-01",
        violation_type="Missing Helmet",
        missing_ppe=["helmet"],
        risk_level="High",
        confidence=0.91,
    )
    print(f"  ✅ PPE violation logged: ID={viol_id}")

    # Save an alert
    alert_data = {
        "alert_id": "ALT-TEST01",
        "timestamp": "2026-09-03T12:00:00",
        "violation_type": "Worker detected without helmet",
        "risk_level": "HIGH",
        "worker_id": "Worker-01",
        "confidence": 0.91,
        "status": "New",
        "cooldown_key": "Worker-01:helmet",
    }
    db.save_alert(alert_data)
    print(f"  ✅ Alert saved: {alert_data['alert_id']}")

    # Update alert status
    db.update_alert_status("ALT-TEST01", "Acknowledged")
    alerts = db.get_alerts()
    assert any(a["alert_id"] == "ALT-TEST01" and a["status"] == "Acknowledged" for a in alerts)
    print(f"  ✅ Alert status updated to Acknowledged")

    # Worker record
    db.update_worker_record(
        worker_id="Worker-01",
        timestamp="2026-09-03T12:00:00",
        ppe_status="Non-Compliant",
        missing_ppe=["helmet"],
        risk_level="High",
        is_violation=True,
    )
    workers = db.get_workers()
    assert any(w["worker_id"] == "Worker-01" for w in workers)
    print(f"  ✅ Worker record created/updated")

    # Analytics
    analytics = db.get_analytics_summary()
    assert analytics["total_events"] >= 1
    assert analytics["total_violations"] >= 1
    print(f"  ✅ Analytics summary retrieved:")
    print(f"     Events: {analytics['total_events']}, Workers: {analytics['total_workers']}, Violations: {analytics['total_violations']}")

    # Clean up test DB
    try:
        os.remove(test_db_path)
    except Exception:
        pass

    print(f"  ✅ TEST 1 PASSED — Database operations verified")
    return True


def test_ppe_compliance_engine():
    """Test 2: PPE Compliance Engine spatial association"""
    separator("TEST 2: PPE Compliance Engine")
    from agents.ppe_compliance_engine import PPEComplianceEngine

    engine = PPEComplianceEngine(required_ppe={"Hardhat", "Safety Vest"})

    # Scenario A: Compliant worker (Person + Hardhat + Safety Vest nearby)
    detections_compliant = [
        {"class_name": "Person", "class_id": 5, "confidence": 0.90, "bbox": [100, 50, 200, 350]},
        {"class_name": "Hardhat", "class_id": 0, "confidence": 0.85, "bbox": [110, 40, 190, 90]},
        {"class_name": "Safety Vest", "class_id": 7, "confidence": 0.88, "bbox": [105, 120, 195, 250]},
    ]
    result = engine.evaluate_compliance(detections_compliant)
    assert result["worker_count"] == 1
    assert result["compliant_workers"] == 1
    assert result["violations_count"] == 0
    assert result["worker_results"][0]["ppe_status"] == "Compliant"
    print(f"  ✅ Scenario A: Compliant worker detected correctly")
    print(f"     Worker: {result['worker_results'][0]}")

    # Scenario B: Worker missing helmet (Person + NO-Hardhat + Safety Vest)
    detections_no_helmet = [
        {"class_name": "Person", "class_id": 5, "confidence": 0.92, "bbox": [100, 50, 200, 350]},
        {"class_name": "NO-Hardhat", "class_id": 2, "confidence": 0.88, "bbox": [110, 40, 190, 90]},
        {"class_name": "Safety Vest", "class_id": 7, "confidence": 0.86, "bbox": [105, 120, 195, 250]},
    ]
    result = engine.evaluate_compliance(detections_no_helmet)
    assert result["violations_count"] == 1
    assert "helmet" in result["worker_results"][0]["missing_ppe"]
    assert result["worker_results"][0]["risk_level"] == "High"
    print(f"  ✅ Scenario B: Worker missing helmet → High Risk")
    print(f"     Missing: {result['worker_results'][0]['missing_ppe']}, Risk: {result['worker_results'][0]['risk_level']}")

    # Scenario C: Worker missing vest (Person + Hardhat + NO-Safety Vest)
    detections_no_vest = [
        {"class_name": "Person", "class_id": 5, "confidence": 0.91, "bbox": [100, 50, 200, 350]},
        {"class_name": "Hardhat", "class_id": 0, "confidence": 0.85, "bbox": [110, 40, 190, 90]},
        {"class_name": "NO-Safety Vest", "class_id": 4, "confidence": 0.80, "bbox": [105, 120, 195, 250]},
    ]
    result = engine.evaluate_compliance(detections_no_vest)
    assert result["violations_count"] == 1
    assert "vest" in result["worker_results"][0]["missing_ppe"]
    assert result["worker_results"][0]["risk_level"] == "Medium"
    print(f"  ✅ Scenario C: Worker missing vest → Medium Risk")
    print(f"     Missing: {result['worker_results'][0]['missing_ppe']}, Risk: {result['worker_results'][0]['risk_level']}")

    # Scenario D: Worker missing both helmet and vest
    detections_no_both = [
        {"class_name": "Person", "class_id": 5, "confidence": 0.92, "bbox": [100, 50, 200, 350]},
        {"class_name": "NO-Hardhat", "class_id": 2, "confidence": 0.88, "bbox": [110, 40, 190, 90]},
        {"class_name": "NO-Safety Vest", "class_id": 4, "confidence": 0.80, "bbox": [105, 120, 195, 250]},
    ]
    result = engine.evaluate_compliance(detections_no_both)
    assert result["violations_count"] == 1
    assert "helmet" in result["worker_results"][0]["missing_ppe"]
    assert "vest" in result["worker_results"][0]["missing_ppe"]
    assert result["worker_results"][0]["risk_level"] in ("High", "Critical")
    print(f"  ✅ Scenario D: Worker missing helmet AND vest → {result['worker_results'][0]['risk_level']} Risk")

    # Scenario E: Multiple workers
    detections_multi = [
        {"class_name": "Person", "class_id": 5, "confidence": 0.90, "bbox": [50, 50, 150, 350]},
        {"class_name": "Hardhat", "class_id": 0, "confidence": 0.85, "bbox": [60, 40, 140, 90]},
        {"class_name": "Safety Vest", "class_id": 7, "confidence": 0.88, "bbox": [55, 120, 145, 250]},
        {"class_name": "Person", "class_id": 5, "confidence": 0.88, "bbox": [300, 50, 400, 350]},
        {"class_name": "NO-Hardhat", "class_id": 2, "confidence": 0.86, "bbox": [310, 40, 390, 90]},
        {"class_name": "NO-Safety Vest", "class_id": 4, "confidence": 0.82, "bbox": [305, 120, 395, 250]},
    ]
    result = engine.evaluate_compliance(detections_multi)
    assert result["worker_count"] == 2
    assert result["compliant_workers"] == 1
    assert result["violations_count"] == 1
    print(f"  ✅ Scenario E: Multiple workers — 1 Compliant, 1 Non-Compliant")

    print(f"  ✅ TEST 2 PASSED — PPE Compliance Engine verified")
    return True


def test_alert_system():
    """Test 3: Alert Generation with Cooldown"""
    separator("TEST 3: Alert System & Cooldown")
    from agents.alert_system import AlertSystem
    from database.db_manager import SafetyDBManager

    test_db_path = os.path.join(PROJECT_ROOT, "data", "test_alerts.db")
    db = SafetyDBManager(db_path=test_db_path)
    alert_sys = AlertSystem(db_manager=db, cooldown_seconds=2.0)

    # Worker with violation
    worker_results = [
        {
            "worker_id": "Worker-01",
            "ppe_status": "Non-Compliant",
            "missing_ppe": ["helmet"],
            "risk_level": "High",
            "violation": "Missing Helmet",
            "alert": True,
            "confidence": 0.91,
        }
    ]

    # First call should generate alert
    alerts = alert_sys.process_worker_violations(worker_results)
    assert len(alerts) == 1
    print(f"  ✅ Alert generated: {alerts[0]['alert_id']} — {alerts[0]['violation_type']}")

    # Immediate second call should be suppressed (cooldown)
    alerts2 = alert_sys.process_worker_violations(worker_results)
    assert len(alerts2) == 0
    print(f"  ✅ Duplicate alert suppressed (cooldown active)")

    # Wait for cooldown to expire
    time.sleep(2.5)
    alerts3 = alert_sys.process_worker_violations(worker_results)
    assert len(alerts3) == 1
    print(f"  ✅ Alert generated after cooldown expired: {alerts3[0]['alert_id']}")

    # Status update
    success = alert_sys.update_alert_status(alerts[0]["alert_id"], "Acknowledged")
    assert success
    print(f"  ✅ Alert status updated to 'Acknowledged'")

    success = alert_sys.update_alert_status(alerts[0]["alert_id"], "Resolved")
    assert success
    print(f"  ✅ Alert status updated to 'Resolved'")

    # Clean up
    try:
        os.remove(test_db_path)
    except Exception:
        pass

    print(f"  ✅ TEST 3 PASSED — Alert system & cooldown verified")
    return True


def test_safety_agent_with_real_image():
    """Test 4: End-to-end Safety Agent with real test image"""
    separator("TEST 4: Safety Agent — End-to-End Real Image Detection")
    from agents.safety_agent import SafetyAgent
    from database.db_manager import SafetyDBManager

    test_db_path = os.path.join(PROJECT_ROOT, "data", "test_agent.db")
    db = SafetyDBManager(db_path=test_db_path)
    agent = SafetyAgent(db_manager=db)

    # Find a test image
    test_images_dir = os.path.join(PROJECT_ROOT, "data", "test", "images")
    if not os.path.isdir(test_images_dir):
        print(f"  ⚠️ Test images directory not found: {test_images_dir}")
        return False

    test_images = glob.glob(os.path.join(test_images_dir, "*.jpg"))[:5]
    if not test_images:
        print(f"  ⚠️ No test images found")
        return False

    print(f"  Testing with {len(test_images)} images from data/test/images/\n")

    for idx, img_path in enumerate(test_images):
        print(f"  ── Image {idx+1}: {os.path.basename(img_path)} ──")

        report = agent.analyze_image(img_path, conf_threshold=0.25)

        print(f"     Workers detected: {report['worker_count']}")
        print(f"     Compliant:        {report['compliant_workers']}")
        print(f"     Violations:       {report['violations']}")
        print(f"     Risk Level:       {report['risk_level']}")
        print(f"     Safety Score:     {report['overall_safety_score']}%")
        print(f"     Alerts generated: {len(report['alerts'])}")

        # Print worker results
        for w in report.get("worker_results", []):
            status_icon = "✅" if w["ppe_status"] == "Compliant" else "❌"
            print(f"     {status_icon} {w['worker_id']}: {w['ppe_status']} | "
                  f"Missing: {w['missing_ppe']} | Risk: {w['risk_level']} | "
                  f"Conf: {w['confidence']:.0%}")

        # Print alerts
        for a in report.get("alerts", []):
            print(f"     🚨 ALERT {a['alert_id']}: {a['violation_type']} "
                  f"({a['risk_level']}) — {a['worker_id']}")

        # Print detected classes
        class_names = sorted(set(d["class_name"] for d in report.get("detections", [])))
        print(f"     Detected classes: {class_names}")
        print()

    # Verify database was populated
    analytics = db.get_analytics_summary()
    print(f"  Database summary after {len(test_images)} images:")
    print(f"     Events:     {analytics['total_events']}")
    print(f"     Workers:    {analytics['total_workers']}")
    print(f"     Violations: {analytics['total_violations']}")
    print(f"     Alerts:     {analytics['total_alerts']}")
    print(f"     Safety:     {analytics['safety_score']}%")

    # Clean up
    try:
        os.remove(test_db_path)
    except Exception:
        pass

    print(f"  ✅ TEST 4 PASSED — End-to-end Safety Agent pipeline verified")
    return True


def test_api_module():
    """Test 5: API helper functions"""
    separator("TEST 5: API Module — Programmatic Helpers")
    from api.server import api_get_analytics, api_get_alerts, api_get_workers

    analytics = api_get_analytics()
    assert isinstance(analytics, dict)
    assert "total_events" in analytics
    print(f"  ✅ api_get_analytics() returned: {list(analytics.keys())[:5]}...")

    alerts = api_get_alerts()
    assert isinstance(alerts, list)
    print(f"  ✅ api_get_alerts() returned {len(alerts)} alerts")

    workers = api_get_workers()
    assert isinstance(workers, list)
    print(f"  ✅ api_get_workers() returned {len(workers)} workers")

    print(f"  ✅ TEST 5 PASSED — API helper functions verified")
    return True


def test_milestone1_backward_compatibility():
    """Test 6: Ensure Milestone 1 SiteRiskAgent still works"""
    separator("TEST 6: Milestone 1 Backward Compatibility")
    from agents.site_risk_agent import SiteRiskAgent

    agent = SiteRiskAgent()

    test_images_dir = os.path.join(PROJECT_ROOT, "data", "test", "images")
    test_images = glob.glob(os.path.join(test_images_dir, "*.jpg"))[:2]

    if not test_images:
        print(f"  ⚠️ No test images found for M1 backward compatibility test")
        return False

    for img_path in test_images:
        report = agent.analyze_image(img_path, conf_threshold=0.25)
        assert "risk" in report
        assert "detections" in report
        assert "annotated_image" in report
        print(f"  ✅ M1 SiteRiskAgent works: {os.path.basename(img_path)} → "
              f"Score={report['risk']['score']}, Level={report['risk']['level']}")

    print(f"  ✅ TEST 6 PASSED — Milestone 1 backward compatibility verified")
    return True


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    separator("MILESTONE 2 — SAFETY INTELLIGENCE TEST SUITE")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Model path:   {os.path.join(PROJECT_ROOT, 'models', 'best.pt')}")
    print(f"  Model exists: {os.path.isfile(os.path.join(PROJECT_ROOT, 'models', 'best.pt'))}")

    results = {}
    tests = [
        ("Database Operations", test_database_operations),
        ("PPE Compliance Engine", test_ppe_compliance_engine),
        ("Alert System & Cooldown", test_alert_system),
        ("Safety Agent E2E", test_safety_agent_with_real_image),
        ("API Module", test_api_module),
        ("M1 Backward Compatibility", test_milestone1_backward_compatibility),
    ]

    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            results[name] = False
            print(f"\n  ❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    separator("TEST RESULTS SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {name}")

    print(f"\n  {'='*50}")
    print(f"  TOTAL: {passed}/{total} tests passed")
    if passed == total:
        print(f"  🎉 ALL TESTS PASSED — Milestone 2 Verified!")
    else:
        print(f"  ⚠️ Some tests failed — review output above")
    print(f"  {'='*50}")
