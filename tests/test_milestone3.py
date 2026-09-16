"""
Milestone 3 — Compliance Agent and Insurance Agent Test Suite
=============================================================
Tests deterministic calculations, explainability, problem-based recommendations,
database persistence with timestamps, and dynamic input responsiveness.
"""

import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agents.compliance_agent import ComplianceAgent
from agents.insurance_agent import InsuranceAgent
from database.db_manager import SafetyDBManager
from risk.construction_risk_engine import ConstructionRiskEngine


def separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def test_example_1_clean_project():
    """Example 1: All Compliant, 0 Violations, 0 Incidents."""
    separator("TEST 1: Example 1 — Fully Compliant Project")
    test_db = SafetyDBManager(db_path=os.path.join(PROJECT_ROOT, "data", "test_m3.db"))
    engine = ConstructionRiskEngine(db_manager=test_db)

    result = engine.assess_project(
        project_id="PROJ-CLEAN",
        inspection_reports=[{"status": "compliant", "finding": "hardhat and safety vest verified"}],
        safety_logs=[{"status": "compliant", "finding": "daily site inspection complete"}],
        incidents=[],
        safety_trends=[{"violations": 0}],
    )

    comp = result["compliance"]
    ins = result["insurance"]

    print(f"  Compliance Score: {comp['overall_compliance_score']}/100 ({comp['compliance_level']})")
    print(f"  Insurance Risk:   {ins['overall_insurance_risk']}/100 ({ins['insurance_risk_level']})")
    print(f"  Recommendations:  {result['recommendations']}")

    assert comp["inspection_score"] == 100.0
    assert comp["safety_log_score"] == 100.0
    assert comp["violation_score"] == 100.0
    assert comp["overall_compliance_score"] == 100.0
    assert comp["compliance_level"] == "Excellent"
    assert comp["regulatory_readiness_score"] == 100.0

    assert ins["incident_risk"] == 0.0
    assert ins["violation_risk"] == 0.0
    assert ins["compliance_risk"] == 0.0
    assert ins["overall_insurance_risk"] == 0.0
    assert ins["insurance_risk_level"] == "Very Low"
    assert ins["insurance_exposure"] == "Minimal"

    assert len(result["recommendations"]) == 1
    assert "No major compliance or insurance risk indicators" in result["recommendations"][0]
    print("  ✅ TEST 1 PASSED: Example 1 matches required scores and outputs")
    return True


def test_example_2_moderate_risk():
    """Example 2: Non-compliant inspection, Non-compliant log, 5 violations, 2 high-severity incidents."""
    separator("TEST 2: Example 2 — Moderate / Non-Compliant Risk")
    test_db = SafetyDBManager(db_path=os.path.join(PROJECT_ROOT, "data", "test_m3.db"))
    engine = ConstructionRiskEngine(db_manager=test_db)

    result = engine.assess_project(
        project_id="PROJ-MOD",
        inspection_reports=[{"status": "non-compliant", "finding": "scaffolding guardrail missing"}],
        safety_logs=[{"status": "non-compliant", "finding": "incomplete site log"}],
        incidents=[
            {"severity": "high", "description": "serious injury on site"},
            {"severity": "high", "description": "worker fall with fracture"},
        ],
        safety_trends=[{"violations": 5}],
    )

    comp = result["compliance"]
    ins = result["insurance"]

    print(f"  Compliance Score: {comp['overall_compliance_score']}/100 ({comp['compliance_level']})")
    print(f"  Insurance Risk:   {ins['overall_insurance_risk']}/100 ({ins['insurance_risk_level']})")
    print(f"  Recommendations:  {result['recommendations']}")

    # Inspection = 60, Log = 60, Violations(5) = 75
    # Overall = (60*0.4) + (60*0.3) + (75*0.3) = 24 + 18 + 22.5 = 64.50
    assert comp["inspection_score"] == 60.0
    assert comp["safety_log_score"] == 60.0
    assert comp["violation_score"] == 75.0
    assert comp["overall_compliance_score"] == 64.50
    assert comp["compliance_level"] == "Moderate"

    # Incidents = 2 high -> incident_risk = 50.0
    # Violations (5) -> violation_risk = 25.0
    # Compliance (64.50 -> 50-74 range) -> compliance_risk = 25.0
    # Insurance = (50.0 * 0.50) + (25.0 * 0.25) + (25.0 * 0.25) = 25 + 6.25 + 6.25 = 37.50
    assert ins["incident_risk"] == 50.0
    assert ins["violation_risk"] == 25.0
    assert ins["compliance_risk"] == 25.0
    assert ins["overall_insurance_risk"] == 37.50
    assert ins["insurance_risk_level"] == "Low"

    # Recommendations must target detected issues
    rec_text = " ".join(result["recommendations"])
    assert "safety violation" in rec_text.lower()
    assert "inspection" in rec_text.lower()
    assert "log" in rec_text.lower()
    assert "incident investigation" in rec_text.lower()

    print("  ✅ TEST 2 PASSED: Example 2 matches required calculations")
    return True


def test_example_3_critical_risk():
    """Example 3: Failed inspection, Failed log, 15 violations, Critical incident."""
    separator("TEST 3: Example 3 — Critical High Risk")
    test_db = SafetyDBManager(db_path=os.path.join(PROJECT_ROOT, "data", "test_m3.db"))
    engine = ConstructionRiskEngine(db_manager=test_db)

    result = engine.assess_project(
        project_id="PROJ-CRIT",
        inspection_reports=[{"status": "failed", "finding": "structural support failure"}],
        safety_logs=[{"status": "failed", "finding": "major safety log failure"}],
        incidents=[{"severity": "critical", "description": "fatal accident / collapse"}],
        safety_trends=[{"violations": 15}],
    )

    comp = result["compliance"]
    ins = result["insurance"]

    print(f"  Compliance Score: {comp['overall_compliance_score']}/100 ({comp['compliance_level']})")
    print(f"  Insurance Risk:   {ins['overall_insurance_risk']}/100 ({ins['insurance_risk_level']})")

    # Inspection = 40, Log = 40, Violations(15) = 25
    # Overall = (40*0.4) + (40*0.3) + (25*0.3) = 16 + 12 + 7.5 = 35.50 (Poor)
    assert comp["inspection_score"] == 40.0
    assert comp["safety_log_score"] == 40.0
    assert comp["violation_score"] == 25.0
    assert comp["overall_compliance_score"] == 35.50
    assert comp["compliance_level"] == "Poor"

    # Incidents = 1 critical (75.0) -> incident_risk = 75.0
    # Violations (15) -> violation_risk = 75.0
    # Compliance (35.50 -> 25-49 range) -> compliance_risk = 50.0
    # Insurance = (75.0 * 0.50) + (75.0 * 0.25) + (50.0 * 0.25) = 37.5 + 18.75 + 12.5 = 68.75 (High)
    assert ins["incident_risk"] == 75.0
    assert ins["violation_risk"] == 75.0
    assert ins["compliance_risk"] == 50.0
    assert ins["overall_insurance_risk"] == 68.75
    assert ins["insurance_risk_level"] == "High"
    assert ins["insurance_exposure"] == "Significant"

    print("  ✅ TEST 3 PASSED: Example 3 matches required calculations")
    return True


def test_dynamic_variation():
    """Verify modifying input dynamically changes output."""
    separator("TEST 4: Dynamic Variation Test")
    test_db = SafetyDBManager(db_path=os.path.join(PROJECT_ROOT, "data", "test_m3.db"))
    engine = ConstructionRiskEngine(db_manager=test_db)

    score_1 = engine.assess_project(
        project_id="PROJ-A",
        inspection_reports=[{"status": "compliant"}],
        safety_logs=[{"status": "compliant"}],
        incidents=[],
        safety_trends=[{"violations": 0}],
    )["compliance"]["overall_compliance_score"]

    score_2 = engine.assess_project(
        project_id="PROJ-A",
        inspection_reports=[{"status": "violation"}],
        safety_logs=[{"status": "failed"}],
        incidents=[],
        safety_trends=[{"violations": 8}],
    )["compliance"]["overall_compliance_score"]

    assert score_1 != score_2
    assert score_1 > score_2
    print(f"  Score 1 (Clean): {score_1} vs Score 2 (Degraded): {score_2}")
    print("  ✅ TEST 4 PASSED: Dynamic variation verified")
    return True


def test_database_persistence_and_timestamp():
    """Verify database persistence and timestamp constraint on insurance_cases."""
    separator("TEST 5: Database Timestamp & Persistence")
    db_path = os.path.join(PROJECT_ROOT, "data", "test_m3.db")
    test_db = SafetyDBManager(db_path=db_path)
    engine = ConstructionRiskEngine(db_manager=test_db)

    result = engine.assess_project(
        project_id="PROJ-DBTEST",
        inspection_reports=[{"status": "compliant"}],
        safety_logs=[{"status": "compliant"}],
        incidents=[{"severity": "medium", "description": "minor equipment damage"}],
        safety_trends=[{"violations": 1}],
    )

    cases = test_db.get_insurance_cases(project_id="PROJ-DBTEST")
    assert len(cases) >= 1
    case = cases[0]
    assert case["timestamp"] is not None
    assert len(case["timestamp"]) > 10
    print(f"  Saved insurance case ID: {case['case_id']}, Timestamp: {case['timestamp']}")

    checks = test_db.get_compliance_checks(project_id="PROJ-DBTEST")
    assert len(checks) >= 1
    print(f"  Saved compliance checks count: {len(checks)}")

    print("  ✅ TEST 5 PASSED: Database persistence and timestamp verified")
    return True


if __name__ == "__main__":
    separator("MILESTONE 3 TEST SUITE")
    tests = [
        ("Example 1 (Clean Project)", test_example_1_clean_project),
        ("Example 2 (Moderate Risk)", test_example_2_moderate_risk),
        ("Example 3 (Critical Risk)", test_example_3_critical_risk),
        ("Dynamic Variation", test_dynamic_variation),
        ("Database Persistence & Timestamps", test_database_persistence_and_timestamp),
    ]

    all_passed = True
    for name, fn in tests:
        try:
            passed = fn()
            if not passed:
                all_passed = False
        except Exception as e:
            all_passed = False
            print(f"  ❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    separator("SUMMARY")
    if all_passed:
        print("  🎉 ALL MILESTONE 3 TESTS PASSED!")
    else:
        print("  ⚠️ SOME TESTS FAILED")

    # Clean up test db
    try:
        p = os.path.join(PROJECT_ROOT, "data", "test_m3.db")
        if os.path.exists(p):
            os.remove(p)
    except Exception:
        pass
