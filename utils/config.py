# Construction Risk Intelligence Platform - Configuration
# ======================================================
# All weights, thresholds, and settings in one place.
# Modify values here to tune the system — nothing is hardcoded elsewhere.
#
# NOTE: Risk weights and thresholds are a PROTOTYPE scoring methodology
# designed for academic demonstration. They are NOT scientifically validated
# real-world safety standards.

import os
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths (all relative to the project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best.pt")
DATA_YAML_PATH = os.path.join(PROJECT_ROOT, "data", "data.yaml")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
LOG_FILE = os.path.join(OUTPUT_DIR, "analysis_log.json")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "safety_intelligence.db")
SAMPLE_IMAGES_DIR = os.path.join(PROJECT_ROOT, "sample_images")

# ---------------------------------------------------------------------------
# Milestone 2 Safety Intelligence Settings
# ---------------------------------------------------------------------------
DEFAULT_REQUIRED_PPE = {"Hardhat", "Safety Vest"}
PERSON_CONFIDENCE_THRESHOLD = 0.50
PPE_CONFIDENCE_THRESHOLD = 0.40
NMS_IOU_THRESHOLD = 0.50
ASSOCIATION_SCORE_THRESHOLD = 0.45
DEFAULT_IOU_THRESHOLD = NMS_IOU_THRESHOLD
ALERT_COOLDOWN_SECONDS = 10.0
ALERT_SEVERITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
ALERT_STATUSES = ["New", "Acknowledged", "Resolved"]

# Fail-safe PPE enforcement: When True, any detected worker lacking positive PPE
# (e.g. Hardhat or Safety Vest) on their body region is classified as Non-Compliant
# rather than Uncertain.
FAIL_SAFE_PPE_ENFORCEMENT = True


# ---------------------------------------------------------------------------
# YOLO class names (order must match the training data.yaml)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "Hardhat",        # 0
    "Mask",           # 1
    "NO-Hardhat",     # 2
    "NO-Mask",        # 3
    "NO-Safety Vest", # 4
    "Person",         # 5
    "Safety Cone",    # 6
    "Safety Vest",    # 7
    "machinery",      # 8
    "vehicle",        # 9
]

# Semantic groupings
HAZARD_CLASSES = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}
SAFETY_EQUIPMENT_CLASSES = {"Hardhat", "Mask", "Safety Vest", "Safety Cone"}
WORKER_CLASSES = {"Person"}
HEAVY_OBJECTS = {"machinery", "vehicle"}

# ---------------------------------------------------------------------------
# Detection settings
# ---------------------------------------------------------------------------
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}

# ---------------------------------------------------------------------------
# Risk weights  (prototype methodology — not validated safety standards)
# ---------------------------------------------------------------------------
HAZARD_WEIGHTS = {
    "NO-Hardhat":       25,  # Moderate single PPE violation
    "NO-Safety Vest":   25,  # Moderate single PPE violation
    "NO-Mask":          5,   # Minor single PPE violation
}

# Proximity-based hazard weights (machinery/vehicle near workers)
PROXIMITY_WEIGHTS = {
    "machinery": 20,
    "vehicle":   20,
}

# When ≥ MULTI_HAZARD_THRESHOLD distinct hazard *types* co-occur,
# the raw score is multiplied by MULTI_HAZARD_MULTIPLIER.
MULTI_HAZARD_THRESHOLD = 3
MULTI_HAZARD_MULTIPLIER = 1.3

# Maximum raw score before normalisation (used to map into 0-100).
# Calibrated so multi-PPE hazards in videos reach High/Critical Risk while compliant scenes remain Low Risk.
MAX_RAW_SCORE = 65


# ---------------------------------------------------------------------------
# Risk levels
# ---------------------------------------------------------------------------
RISK_LEVELS = [
    {"min": 0,  "max": 24,  "label": "Low",      "color": "#22c55e", "emoji": "🟢"},
    {"min": 25, "max": 49,  "label": "Medium",    "color": "#f59e0b", "emoji": "🟡"},
    {"min": 50, "max": 74,  "label": "High",      "color": "#f97316", "emoji": "🟠"},
    {"min": 75, "max": 100, "label": "Critical",  "color": "#ef4444", "emoji": "🔴"},
]

# ---------------------------------------------------------------------------
# Bounding-box proximity
# ---------------------------------------------------------------------------
# Two objects are considered "near each other" when the distance between
# the centres of their bounding boxes is ≤ PROXIMITY_DISTANCE_RATIO × the
# average of their bounding-box diagonals. 
PROXIMITY_DISTANCE_RATIO = 1.5

# ---------------------------------------------------------------------------
# Visualisation colours  (BGR for OpenCV)
# ---------------------------------------------------------------------------
HAZARD_BOX_COLOR = (0, 0, 255)       # Red
SAFETY_BOX_COLOR = (0, 200, 0)       # Green
WORKER_BOX_COLOR = (255, 165, 0)     # Orange
HEAVY_BOX_COLOR = (0, 140, 255)      # Dark orange
DEFAULT_BOX_COLOR = (200, 200, 200)  # Grey
BOX_THICKNESS = 2
FONT_SCALE = 0.5

# ---------------------------------------------------------------------------
# Dashboard text
# ---------------------------------------------------------------------------
APP_TITLE = "Construction Risk Intelligence Dashboard"
APP_SUBTITLE = "AI-Powered Site Risk Monitoring & Hazard Detection"

# ---------------------------------------------------------------------------
# Recommendations templates (keyed by hazard type)
# ---------------------------------------------------------------------------
RECOMMENDATIONS = {
    "NO-Hardhat": "⛑️ Ensure all workers wear approved hard hats before entering the site.",
    "NO-Mask": "😷 Enforce face-mask requirements in accordance with site safety policy.",
    "NO-Safety Vest": "🦺 All personnel must wear high-visibility safety vests at all times.",
    "ppe_compliance_gap": "🦺 Verify that all workers wear hardhats, high-visibility vests, and safety footwear.",
    "machinery_proximity": "🚜 Restrict worker access around active machinery — maintain a safe exclusion zone.",
    "vehicle_proximity": "🚛 Maintain safe separation between workers and moving vehicles on site.",
    "multiple_hazards": "🚨 Multiple hazards detected — conduct an immediate site safety inspection.",
    "high_risk": "⚠️ Site risk level is HIGH or CRITICAL — consider halting non-essential work until issues are resolved.",
}

# Fallback when no hazards are found
NO_HAZARD_MESSAGE = "✅ No significant hazards detected in the analysed frame."

# ---------------------------------------------------------------------------
# SMS Alert & Emergency Dispatch Settings
# ---------------------------------------------------------------------------
SMS_ENABLED = True
SMS_TRIGGER_LEVELS = {"HIGH", "CRITICAL"}
DEFAULT_SUPERVISOR_PHONE = "+91 6370671276"
DEFAULT_WORKER_PHONES = {
    "Worker-01": "+91 6370671276",
    "Worker-02": "+91 6370671276",
    "Worker-03": "+91 6370671276",
    "Worker-04": "+91 6370671276",
    "Worker-05": "+91 6370671276",
    "Worker-06": "+91 6370671276",
    "Worker-07": "+91 6370671276",
}

# ---------------------------------------------------------------------------
# Milestone 4 — Reporting Intelligence & Enterprise Deployment
# ---------------------------------------------------------------------------

# Global Risk Scoring Weights (must sum to 1.0)
GLOBAL_RISK_WEIGHTS = {
    "active_hazards": 0.25,
    "ppe_compliance": 0.25,
    "regulatory_violations": 0.25,
    "insurance_exposure": 0.25,
}

# Agent Performance Baseline Thresholds (%)
AGENT_PERFORMANCE_BASELINES = {
    "site_risk_agent": {"efficiency": 96.0, "accuracy": 94.0},
    "safety_agent": {"efficiency": 94.0, "accuracy": 94.0},
    "compliance_agent": {"efficiency": 95.0, "accuracy": 95.0},
    "insurance_agent": {"efficiency": 92.0, "accuracy": 92.0},
}

# Predictive Analytics Cost Model
COST_PER_INCIDENT_PREVENTED = 35000.0   # USD estimated cost savings per prevented incident
BASELINE_INCIDENT_RATE = 0.12           # Historical incidents per worker per month
COMPLIANCE_IMPROVEMENT_BASELINE = 72.0  # Baseline compliance % before platform deployment

# Escalation Workflow Configuration
ESCALATION_RULES = {
    "LOW":      ["log"],
    "MEDIUM":   ["log", "email"],
    "HIGH":     ["log", "email", "sms"],
    "CRITICAL": ["log", "email", "sms", "webhook"],
}

# SMTP Email Settings (loaded from .env)
SMTP_HOST = os.environ.get("SMTP_HOST", "localhost")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM_EMAIL = os.environ.get("SMTP_FROM_EMAIL", "alerts@construction-risk.local")
ESCALATION_EMAIL_RECIPIENTS = [
    e.strip() for e in os.environ.get("ESCALATION_EMAIL_RECIPIENTS", "").split(",") if e.strip()
]

# Slack / Microsoft Teams Webhook URLs (loaded from .env)
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")
TEAMS_WEBHOOK_URL = os.environ.get("TEAMS_WEBHOOK_URL", "")

# Report Configuration
REPORT_TYPES = ["daily_operational", "executive_summary", "audit_report", "project_health"]
REPORT_RETENTION_DAYS = 90

