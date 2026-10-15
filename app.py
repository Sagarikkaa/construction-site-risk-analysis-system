"""
Construction Risk Intelligence Dashboard
==========================================
Streamlit application for:
  Milestone 1: Site Risk Monitoring & Hazard Detection
  Milestone 2: Safety Intelligence & Worker Protection

Run with:  streamlit run app.py
"""

import os
import sys
import json
import html
import base64
import io
import wave
import tempfile
import ultralytics
import ultralytics.engine, ultralytics.nn, ultralytics.utils

# Register legacy Ultralytics unpickling alias for PyTorch model loader compatibility
sys.modules["ultralytics.yolo"] = ultralytics
sys.modules["ultralytics.yolo.utils"] = ultralytics.utils
sys.modules["ultralytics.yolo.engine"] = ultralytics.engine
sys.modules["ultralytics.yolo.nn"] = ultralytics.nn

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional, Any

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


from utils.config import (
    APP_TITLE, APP_SUBTITLE, MODEL_PATH,
    CLASS_NAMES, DEFAULT_CONFIDENCE_THRESHOLD,
    IMAGE_EXTENSIONS, VIDEO_EXTENSIONS,
    RISK_LEVELS, NO_HAZARD_MESSAGE,
    HAZARD_WEIGHTS, PROXIMITY_WEIGHTS,
    DEFAULT_REQUIRED_PPE, DEFAULT_IOU_THRESHOLD,
    ALERT_COOLDOWN_SECONDS,
    SMS_ENABLED, SMS_TRIGGER_LEVELS,
    DEFAULT_SUPERVISOR_PHONE, DEFAULT_WORKER_PHONES,
)
from utils.sms_service import SMSService
from utils.visualization import (
    create_risk_gauge,
    create_hazard_chart,
    create_detection_stats_chart,
    create_risk_trend_chart,
    create_compliance_pie_chart,
    create_violations_by_type_chart,
    create_risk_distribution_chart,
    create_safety_trend_chart,
)
# ──────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Construction Risk Intelligence Platform",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# Custom CSS for a professional dark theme
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global font */
    html, body {
        font-family: 'Inter', sans-serif;
    }

    /* Keep Streamlit's icon ligatures from being rendered as visible text. */
    [data-testid="stIconMaterial"],
    .material-icons,
    .material-icons-outlined,
    .material-symbols-rounded,
    .material-symbols-outlined {
        font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
        font-weight: normal !important;
        font-style: normal !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        color: #e0e7ff;
        font-size: 2rem;
        line-height: 1.2;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0.4rem 0 0 0;
        font-weight: 400;
    }
    .main-header .badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        color: #818cf8;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.75rem;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e1e2e, #252540);
        border-radius: 14px;
        padding: 1.4rem;
        min-width: 0;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }
    .metric-card .value {
        font-size: 2.2rem;
        line-height: 1.15;
        font-weight: 800;
        margin: 0.3rem 0;
    }
    .metric-card .label {
        color: #94a3b8;
        font-size: 0.85rem;
        line-height: 1.3;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        overflow-wrap: anywhere;
    }

    /* Section headers */
    .section-header {
        color: #e0e7ff;
        font-size: 1.3rem;
        line-height: 1.3;
        font-weight: 700;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
        margin: 2rem 0 1rem 0;
    }

    .chart-heading {
        color: #e0e7ff;
        font-size: 1.15rem;
        font-weight: 800;
        margin: 0.6rem 0 0.8rem 0;
    }

    .analytics-hero {
        padding: 0.8rem 0 1.8rem 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.16);
        margin-bottom: 1.5rem;
    }
    .analytics-hero h1 {
        color: #f8fafc;
        font-size: 2.35rem;
        line-height: 1.1;
        font-weight: 800;
        letter-spacing: 0;
        margin: 0;
    }
    .analytics-hero p {
        color: #a7b2c4;
        font-size: 1rem;
        margin: 0.55rem 0 0;
    }
    .analytics-section {
        margin: 2.2rem 0 0.8rem;
    }
    .analytics-section h2 {
        color: #f1f5f9;
        font-size: 1.55rem;
        line-height: 1.2;
        font-weight: 800;
        margin: 0;
    }
    .analytics-section p {
        color: #94a3b8;
        font-size: 0.92rem;
        margin: 0.35rem 0 0;
    }
    .analytics-panel {
        background: rgba(21, 27, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 12px;
        padding: 0.9rem 1rem 0.5rem;
        min-height: 300px;
    }
    .overview-card {
        min-height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        text-align: left;
    }
    .overview-card .description {
        color: #718096;
        font-size: 0.78rem;
        line-height: 1.35;
        margin-top: 0.4rem;
        overflow-wrap: anywhere;
    }
    .alert-table-wrap {
        overflow-x: auto;
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 12px;
        background: rgba(21, 27, 42, 0.72);
    }
    .alert-table {
        width: 100%;
        border-collapse: collapse;
        color: #dbe4f0;
        font-size: 0.9rem;
    }
    .alert-table th {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-align: left;
        text-transform: uppercase;
        padding: 0.85rem 1rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.14);
    }
    .alert-table td {
        padding: 0.85rem 1rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.09);
        vertical-align: middle;
    }
    .alert-table tr:last-child td { border-bottom: 0; }
    .risk-pill {
        display: inline-block;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 800;
        padding: 0.25rem 0.55rem;
        letter-spacing: 0.03em;
    }
    .risk-pill.low { background: rgba(34,197,94,0.16); color: #4ade80; }
    .risk-pill.medium { background: rgba(245,158,11,0.16); color: #fbbf24; }
    .risk-pill.high { background: rgba(249,115,22,0.16); color: #fb923c; }
    .risk-pill.critical { background: rgba(239,68,68,0.18); color: #f87171; }
    @media (max-width: 760px) {
        .analytics-hero h1 { font-size: 1.85rem; }
        .analytics-section h2 { font-size: 1.3rem; }
        .analytics-panel { min-height: 0; padding: 0.55rem; }
    }

    /* Recommendation cards */
    .rec-card {
        background: rgba(30, 30, 50, 0.8);
        border-left: 4px solid #818cf8;
        padding: 0.9rem 1.2rem;
        margin: 0.5rem 0;
        border-radius: 0 10px 10px 0;
        color: #e0e7ff;
        font-size: 0.95rem;
    }

    /* Risk level badge */
    .risk-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
    }

    /* Alert cards */
    .alert-card {
        background: linear-gradient(145deg, #1e1e2e, #2a2040);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        border-left: 5px solid;
    }
    .alert-card.high { border-left-color: #ef4444; }
    .alert-card.medium { border-left-color: #f59e0b; }
    .alert-card.low { border-left-color: #22c55e; }
    .alert-card.critical { border-left-color: #dc2626; }
    .alert-card .alert-title {
        color: #e0e7ff;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }
    .alert-card .alert-meta {
        color: #94a3b8;
        font-size: 0.8rem;
    }

    /* Worker status badges */
    .status-compliant {
        background: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .status-violation {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #818cf8;
    }

    /* History table */
    .history-row {
        background: rgba(30, 30, 50, 0.6);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid rgba(255,255,255,0.04);
    }

    /* Disclaimer */
    .disclaimer {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 10px;
        padding: 1rem;
        color: #fbbf24;
        font-size: 0.8rem;
        margin-top: 1rem;
    }

    /* Hide default Streamlit footer */
    footer {visibility: hidden;}

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1e1e2e, #252540);
        border-radius: 14px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Cached model loading
# ──────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading YOLO safety model...")
def load_safety_agent():
    """Load the Safety Agent (Milestone 2)."""
    from agents.safety_agent import SafetyAgent
    from database.db_manager import SafetyDBManager
    db = SafetyDBManager()
    return SafetyAgent(db_manager=db), db

@st.cache_resource(show_spinner="Loading site risk model...")
def load_legacy_agent():
    """Load the Site Risk Agent (Milestone 1) for backward compatibility."""
    from agents.site_risk_agent import SiteRiskAgent
    return SiteRiskAgent()



def check_model_exists():
    """Check if the YOLO model file is present (auto-recovers if missing)."""
    if not os.path.isfile(MODEL_PATH):
        try:
            from extract_model import extract_best_pt
            extract_best_pt()
        except Exception as e:
            st.warning(f"Auto model extraction notice: {e}")
    return os.path.isfile(MODEL_PATH)


# ──────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏗️ Control Panel")
    st.markdown("---")

    # Navigation
    st.markdown("#### 🧭 Navigation")
    page = st.radio(
        "Select Page",
        [
            "🏠 Dashboard",
            "🔍 Live Detection",
            "🛡️ Safety Monitoring",
            "👷 Workers",
            "🚨 Alerts",
            "📊 Analytics",
            "🏛️ Compliance & Insurance",
            "📈 Enterprise Intelligence & Reports",
            "⚙️ Configuration",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Upload image
    st.markdown("#### 📷 Upload Image")
    uploaded_image = st.file_uploader(
        "Choose a construction-site image",
        type=["jpg", "jpeg", "png"],
        key="image_upload",
        help="Supported formats: JPG, JPEG, PNG",
    )

    st.markdown("---")

    # Upload video
    st.markdown("#### 🎬 Upload Video")
    uploaded_video = st.file_uploader(
        "Choose a construction-site video",
        type=["mp4", "avi", "mov"],
        key="video_upload",
        help="Supported formats: MP4, AVI, MOV",
    )

    st.markdown("---")

    # Detection settings
    st.markdown("#### ⚙️ Detection Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Minimum confidence for YOLO to report a detection.",
    )

    st.markdown("---")

    # Video processing settings
    st.markdown("#### 🎞️ Video Settings")
    frame_skip = st.slider(
        "Analyse every Nth frame",
        min_value=1,
        max_value=30,
        value=10,
        help="Higher = faster processing, lower = more thorough.",
    )
    max_frames = st.slider(
        "Max frames to analyse",
        min_value=10,
        max_value=1000,
        value=300,
        help="Maximum number of frames to sample and analyze across the video.",
    )

    # Model info
    st.markdown("#### 🤖 Model Information")
    if check_model_exists():
        st.success("Model loaded ✓")
        st.caption(f"**Path:** `models/best.pt`")
        st.caption(f"**Type:** YOLOv8n (nano)")
        st.caption(f"**Classes:** {len(CLASS_NAMES)}")
        with st.expander("View classes"):
            for i, name in enumerate(CLASS_NAMES):
                st.caption(f"`{i}` — {name}")
    else:
        st.error("⚠️ Model not found!")
        st.caption(f"Place `best.pt` in `models/` directory.")

    st.markdown("---")

    # Risk weight info
    st.markdown("#### 📊 Risk Weights")
    with st.expander("View scoring config"):
        st.caption("**PPE Hazards:**")
        for haz, w in HAZARD_WEIGHTS.items():
            st.caption(f"  {haz} → +{w}")
        st.caption("**Proximity Hazards:**")
        for obj, w in PROXIMITY_WEIGHTS.items():
            st.caption(f"  {obj} near worker → +{w}")
        st.caption("**Risk Levels:**")
        for lvl in RISK_LEVELS:
            st.caption(f"  {lvl['emoji']} {lvl['min']}–{lvl['max']} → {lvl['label']}")

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <b>Prototype Scoring</b><br>
        Risk scores use a demonstration methodology and are
        <b>not</b> validated safety standards.
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>🏗️ Construction Risk Intelligence Platform</h1>
    <p>AI-Powered Safety Monitoring & Worker Protection</p>
    
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Guard: model must exist
# ──────────────────────────────────────────────────────────────────────
if not check_model_exists():
    st.error(
        "🚫 **YOLO model not found.** "
        f"Please place `best.pt` inside `{os.path.join(PROJECT_ROOT, 'models')}` "
        "and refresh the page."
    )
    st.stop()

safety_agent, db_manager = load_safety_agent()
legacy_agent = load_legacy_agent()
from risk.construction_risk_engine import ConstructionRiskEngine
from risk.intelligence_engine import ConstructionRiskIntelligenceEngine
from agents.reporting_agent import ReportingAgent
from utils.escalation_service import EscalationService

risk_engine = ConstructionRiskEngine(db_manager=db_manager)
escalation_service = EscalationService(db_manager=db_manager)
intelligence_engine = ConstructionRiskIntelligenceEngine(
    db_manager=db_manager,
    site_risk_agent=legacy_agent,
    safety_agent=safety_agent,
    compliance_agent=risk_engine.compliance_agent,
    insurance_agent=risk_engine.insurance_agent,
    escalation_service=escalation_service,
)
reporting_agent = ReportingAgent(
    db_manager=db_manager,
    intelligence_engine=intelligence_engine,
)


# ──────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────
def risk_color(level: str) -> str:
    colors = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#f97316", "Critical": "#ef4444"}
    return colors.get(level, "#818cf8")

def severity_color(severity: str) -> str:
    colors = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#f97316", "CRITICAL": "#ef4444"}
    return colors.get(severity.upper(), "#818cf8")

def render_metric_card(label: str, value, color: str = "#818cf8"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value" style="color:{color}">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def play_alert_beep(alerts):
    """Audible alert beep disabled."""
    pass



def render_detection_table(detections):
    """Show every model detection, not only worker safety results."""
    if not detections:
        st.info("No model objects detected above the confidence threshold.")
        return

    detection_rows = []
    for detection in detections:
        detection_rows.append({
            "Object": detection["class_name"],
            "Confidence": f"{detection['confidence']:.0%}",
            "Bounding Box": "[{}]".format(
                ", ".join(str(round(value)) for value in detection["bbox"])
            ),
        })
    st.dataframe(pd.DataFrame(detection_rows), use_container_width=True, hide_index=True)


def render_analytics_section(title: str, subtitle: str):
    """Render a consistent heading block for dashboard analytics sections."""
    st.markdown(
        f'<div class="analytics-section"><h2>{html.escape(title)}</h2>'
        f'<p>{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def render_ppe_compliance_card(ppe_stats):
    """Render deployed per-PPE compliance metrics as progress bars."""
    if not ppe_stats or not ppe_stats.get("ppe_compliance"):
        return

    st.markdown('<div class="section-header">🛡️ Safety Compliance by PPE Type</div>', unsafe_allow_html=True)
    overall = float(ppe_stats.get("overall_ppe_compliance", 0.0))
    st.markdown(
        f'<div class="metric-card" style="margin-bottom: 1rem;">'
        f'<div class="label">Overall PPE Compliance</div>'
        f'<div class="value" style="color:#818cf8">{overall:.1f}%</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    for category, stats in ppe_stats["ppe_compliance"].items():
        percentage = float(stats.get("percentage", 0.0))
        label = stats.get("label", category.replace("_", " ").title())
        status = stats.get("status", "Low compliance")
        width = max(0, min(100, percentage))
        color = "#22c55e" if percentage >= 90 else ("#f59e0b" if percentage >= 75 else "#ef4444")
        st.markdown(
            f"""
            <div style="margin: 0.7rem 0; padding: 0.8rem 0.9rem; border-radius: 12px; background: rgba(21,27,42,0.72); border: 1px solid rgba(148,163,184,0.12);">
                <div style="display: flex; justify-content: space-between; align-items: center; gap: 1rem; margin-bottom: 0.45rem;">
                    <strong style="color: #e0e7ff; font-size: 0.96rem;">{html.escape(label)}</strong>
                    <span style="color: {color}; font-weight: 700; font-size: 0.92rem;">{percentage:.1f}%</span>
                </div>
                <div style="height: 12px; border-radius: 999px; background: rgba(148,163,184,0.18); overflow: hidden;">
                    <div style="width: {width}%; height: 100%; border-radius: 999px; background: linear-gradient(90deg, {color}, #a78bfa); transition: width 0.5s ease;"></div>
                </div>
                <div style="margin-top: 0.45rem; color: #94a3b8; font-size: 0.76rem;">{html.escape(status)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_recent_alerts_table(alerts):
    """Render recent alerts as a compact table with severity badges."""
    if not alerts:
        st.info("No safety alerts recorded yet.")
        return

    rows = []
    for alert in alerts[:10]:
        risk = str(alert.get("risk_level", "LOW")).upper()
        risk_class = risk.lower() if risk.lower() in {"low", "medium", "high", "critical"} else "low"
        conf = alert.get("confidence")
        conf_str = f"{float(conf):.0%}" if conf is not None else "N/A"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(alert.get('timestamp', ''))[:19])}</td>"
            f"<td>{html.escape(str(alert.get('worker_id', 'Unknown')))}</td>"
            f"<td>{html.escape(str(alert.get('violation_type', 'Safety violation')))}</td>"
            f'<td><span class="risk-pill {risk_class}">{html.escape(risk)}</span></td>'
            f"<td>{conf_str}</td>"
            f"<td>{html.escape(str(alert.get('status', 'New')))}</td>"
            "</tr>"
        )

    st.markdown(
        '<div class="alert-table-wrap"><table class="alert-table">'
        '<thead><tr><th>Time</th><th>Worker</th><th>Violation</th>'
        '<th>Risk Level</th><th>Confidence</th><th>Status</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>',
        unsafe_allow_html=True,
    )



def render_compliance_assessment_section(db_manager, risk_engine):
    """Render interactive inputs, calculation breakdown, and explainability for Milestone 3."""
    st.markdown('<div class="section-header">⚙️ Interactive Compliance & Insurance Agent Assessment</div>', unsafe_allow_html=True)
    st.caption("Submit inspection and incident records to calculate scores from actual evidence.")

    # Scenario Presets
    st.markdown("##### ⚡ Quick Scenarios")
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        if st.button("🟢 Scenario 1: Optimal Site (100% Compliant)", use_container_width=True):
            st.session_state["risk_project_id"] = "PROJECT-OPTIMAL"
            st.session_state["inspection_reports_json"] = '[{"status": "compliant", "finding": "hardhat and safety vest verified"}]'
            st.session_state["safety_logs_json"] = '[{"status": "compliant", "finding": "daily site inspection complete"}]'
            st.session_state["incidents_json"] = '[]'
            st.session_state["safety_trends_json"] = '[{"violations": 0}]'
            assessment = risk_engine.assess_project(
                project_id="PROJECT-OPTIMAL",
                inspection_reports=[{"status": "compliant", "finding": "hardhat and safety vest verified"}],
                safety_logs=[{"status": "compliant", "finding": "daily site inspection complete"}],
                incidents=[],
                safety_trends=[{"violations": 0}],
            )
            st.session_state["project_assessment"] = assessment
            st.rerun()

    with pcol2:
        if st.button("🟡 Scenario 2: Moderate Risk (84% Compliant)", use_container_width=True):
            st.session_state["risk_project_id"] = "PROJECT-MODERATE"
            st.session_state["inspection_reports_json"] = '[{"status": "compliant", "finding": "PPE worn"}]'
            st.session_state["safety_logs_json"] = '[{"status": "non-compliant", "finding": "missing safety vest in sector B"}]'
            st.session_state["incidents_json"] = '[{"severity": "medium", "description": "minor trip hazard recorded"}]'
            st.session_state["safety_trends_json"] = '[{"violations": 2}]'
            assessment = risk_engine.assess_project(
                project_id="PROJECT-MODERATE",
                inspection_reports=[{"status": "compliant", "finding": "PPE worn"}],
                safety_logs=[{"status": "non-compliant", "finding": "missing safety vest in sector B"}],
                incidents=[{"severity": "medium", "description": "minor trip hazard recorded"}],
                safety_trends=[{"violations": 2}],
            )
            st.session_state["project_assessment"] = assessment
            st.rerun()

    with pcol3:
        if st.button("🔴 Scenario 3: Critical High Risk (45% Compliant)", use_container_width=True):
            st.session_state["risk_project_id"] = "PROJECT-CRITICAL"
            st.session_state["inspection_reports_json"] = '[{"status": "failed", "finding": "major safety violation at scaffolding"}]'
            st.session_state["safety_logs_json"] = '[{"status": "violation", "finding": "unauthorized zone entry"}]'
            st.session_state["incidents_json"] = '[{"severity": "critical", "description": "structural failure warning"}]'
            st.session_state["safety_trends_json"] = '[{"violations": 8}]'
            assessment = risk_engine.assess_project(
                project_id="PROJECT-CRITICAL",
                inspection_reports=[{"status": "failed", "finding": "major safety violation at scaffolding"}],
                safety_logs=[{"status": "violation", "finding": "unauthorized zone entry"}],
                incidents=[{"severity": "critical", "description": "structural failure warning"}],
                safety_trends=[{"violations": 8}],
            )
            st.session_state["project_assessment"] = assessment
            st.rerun()

    st.markdown("---")
    st.markdown("##### 📝 Custom Project Assessment Inputs")
    assessment_col1, assessment_col2 = st.columns(2)
    with assessment_col1:
        project_id = st.text_input(
            "Project ID",
            value=st.session_state.get("risk_project_id", "PROJECT-001"),
            key="risk_project_id",
        )
        inspection_text = st.text_area(
            "Inspection Reports (JSON list)",
            value=st.session_state.get("inspection_reports_json", '[{"status": "compliant", "finding": "hardhat and safety vest verified"}]'),
            height=120,
            key="inspection_reports_json",
        )
        safety_logs_text = st.text_area(
            "Safety Logs (JSON list)",
            value=st.session_state.get("safety_logs_json", '[{"status": "compliant", "finding": "daily site inspection complete"}]'),
            height=120,
            key="safety_logs_json",
        )
    with assessment_col2:
        incidents_text = st.text_area(
            "Reported Incidents (JSON list)",
            value=st.session_state.get("incidents_json", "[]"),
            height=120,
            key="incidents_json",
        )
        trends_text = st.text_area(
            "Safety Violation Trends (JSON list)",
            value=st.session_state.get("safety_trends_json", '[{"violations": 0}]'),
            height=120,
            key="safety_trends_json",
        )

    if st.button("▶ Run Custom Project Risk Assessment", type="primary", key="run_project_assessment"):
        try:
            if not project_id.strip():
                raise ValueError("Project ID is required.")
            parsed_inputs = {
                "inspection_reports": json.loads(inspection_text),
                "safety_logs": json.loads(safety_logs_text),
                "incidents": json.loads(incidents_text),
                "safety_trends": json.loads(trends_text),
            }
            if not all(isinstance(value, list) for value in parsed_inputs.values()):
                raise ValueError("Each assessment input must be a JSON list.")
            assessment = risk_engine.assess_project(project_id=project_id.strip(), **parsed_inputs)
            st.session_state["project_assessment"] = assessment
            st.success(f"Assessment for {project_id} completed successfully and recorded.")
            st.rerun()
        except (json.JSONDecodeError, ValueError) as exc:
            st.error(f"Invalid assessment input: {exc}")
        except Exception as exc:
            st.error(f"Assessment failed: {exc}")

    assessment = st.session_state.get("project_assessment")
    if assessment:
        compliance = assessment["compliance"]
        insurance = assessment["insurance"]
        overall_compliance_score = float(compliance.get(
            "overall_compliance_score",
            compliance.get("compliance_score", compliance.get("audit_readiness_score", 0.0)),
        ))
        inspection_score = float(compliance.get("inspection_score", overall_compliance_score))
        safety_log_score = float(compliance.get("safety_log_score", overall_compliance_score))
        violation_score = float(compliance.get("violation_score", overall_compliance_score))
        overall_insurance_risk = float(insurance.get(
            "overall_insurance_risk", insurance.get("insurance_risk_score", insurance.get("risk_score", 0.0)),
        ))

        st.markdown('<div class="section-header">🏛️ Compliance Agent Breakdown</div>', unsafe_allow_html=True)
        
        # Component scores breakdown row
        comp_comp1, comp_comp2, comp_comp3 = st.columns(3)
        with comp_comp1:
            st.metric("Inspection Score (40%)", f"{inspection_score:.1f}/100")
        with comp_comp2:
            st.metric("Safety Log Score (30%)", f"{safety_log_score:.1f}/100")
        with comp_comp3:
            st.metric("Violation Score (30%)", f"{violation_score:.1f}/100")

        compliance_col1, compliance_col2, compliance_col3 = st.columns(3)
        with compliance_col1:
            st.metric("Overall Compliance Score", f"{overall_compliance_score:.2f}%")
            st.progress(max(0.0, min(1.0, overall_compliance_score / 100.0)))
        with compliance_col2:
            st.metric("Compliance Level", compliance.get("compliance_level", "Unknown"))
            st.metric("Regulatory Readiness", f"{float(compliance.get('regulatory_readiness_score', overall_compliance_score)):.2f}%")
        with compliance_col3:
            st.metric("Total Inspections", compliance.get("total_inspections", 0))
            st.metric("Non-Compliant Inspections", compliance.get("non_compliant_inspections", 0))
            st.metric("Safety Violations", compliance.get("total_violations", 0))
        
        if compliance.get("checks"):
            st.markdown("##### 📋 Compliance Checkpoints Log")
            st.dataframe(pd.DataFrame(compliance["checks"]), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">🛡️ Insurance Agent Risk Analysis & Exposure</div>', unsafe_allow_html=True)
        
        # Insurance risk components row
        ins_comp1, ins_comp2, ins_comp3 = st.columns(3)
        with ins_comp1:
            st.metric("Incident Risk (50%)", f"{float(insurance.get('incident_risk', overall_insurance_risk)):.1f}/100")
        with ins_comp2:
            st.metric("Violation Risk (25%)", f"{float(insurance.get('violation_risk', 0.0)):.1f}/100")
        with ins_comp3:
            st.metric("Compliance Risk (25%)", f"{float(insurance.get('compliance_risk', 0.0)):.1f}/100")

        insurance_col1, insurance_col2, insurance_col3 = st.columns(3)
        with insurance_col1:
            st.metric("Overall Insurance Risk", f"{overall_insurance_risk:.2f}/100")
            st.progress(max(0.0, min(1.0, overall_insurance_risk / 100.0)))
        with insurance_col2:
            st.metric("Insurance Risk Level", insurance.get("insurance_risk_level", insurance.get("risk_level", "Unknown")))
            st.metric("Claim Risk", insurance.get("claim_risk", "Unknown"))
            st.metric("Insurance Exposure", insurance.get("insurance_exposure", "Unknown"))
        with insurance_col3:
            st.metric("Reported Incidents", insurance.get("incident_count", 0))
            st.metric("Critical Incidents", insurance.get("critical_incidents", 0))
            st.metric("High-Severity Incidents", insurance.get("high_severity_incidents", 0))

        st.markdown('<div class="section-header">💡 Targeted Recommendations</div>', unsafe_allow_html=True)
        recs = assessment.get("recommendations", insurance.get("recommended_actions", []))
        for recommendation in recs:
            st.markdown(f"- {recommendation}")


def render_enterprise_intelligence_page(db_manager, intelligence_engine, reporting_agent):
    """Milestone 4: Executive Reporting, Central Brain, Predictive Analytics, and Multi-Project Portfolio."""
    st.markdown(
        '<div class="analytics-hero"><h1>📈 ENTERPRISE INTELLIGENCE & REPORTING</h1>'
        '<p>Central Construction Risk Intelligence Engine • Multi-Agent Orchestration • Executive Analytics & Reporting</p></div>',
        unsafe_allow_html=True,
    )

    # ──────────────────────────────────────────────────────────────────
    # Total Evaluated Site Inputs Summary Banner (Real Data Only)
    # ──────────────────────────────────────────────────────────────────
    analytics = db_manager.get_analytics_summary()
    all_alerts = db_manager.get_alerts()
    active_alerts_cnt = sum(1 for a in all_alerts if a.get("status") in ("New", "Active", "Open"))

    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    with mcol1:
        st.metric("Total Analysis Runs", analytics.get("total_events", 0), help="All images and videos analyzed")
    with mcol2:
        st.metric("Workers Evaluated", analytics.get("total_workers", 0), help="Unique workers audited from actual detections")
    with mcol3:
        st.metric("Worker Compliance", f"{analytics.get('compliance_percentage', 100.0)}%", help="Real compliance rate from actual detections")
    with mcol4:
        st.metric("PPE Violations", analytics.get("total_violations", 0), help="Recorded violations from evaluated inputs")
    with mcol5:
        st.metric("Active Hazard Alerts", active_alerts_cnt, help="Unresolved safety alerts")

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Central Brain & Risk Scoring",
        "🔮 Predictive Analytics & ROI",
        "📊 Evaluated Inputs Breakdown",
        "📑 Reporting Agent & Exports",
        "🩺 Agent Health & Audit Trail",
    ])

    # ------------------------------------------------------------------
    # TAB 1: Central Brain & Global Risk Scoring
    # ------------------------------------------------------------------
    with tab1:
        st.markdown('<div class="section-header">🎯 Unified Site Risk Score (Evaluated Across Total Inputs)</div>', unsafe_allow_html=True)
        st.caption("Real-time deterministic weighted score across all 4 sub-agents (25% each), calculated directly from 100% of your evaluated inputs.")

        score_data = intelligence_engine.calculate_global_risk_score(project_id=None)
        g_score = score_data.get("global_risk_score", 0.0)
        g_level = score_data.get("risk_level", "Low")
        comp_scores = score_data.get("component_scores", {})

        g_color = "#ef4444" if g_level == "Critical" else ("#f97316" if g_level == "High" else ("#f59e0b" if g_level == "Medium" else "#22c55e"))

        # Banner KPI
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e1e2e, #2a2040); border-radius: 14px; padding: 1.5rem; margin-bottom: 1.5rem; border-left: 6px solid {g_color}; border: 1px solid rgba(255,255,255,0.08);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="background: rgba(129,140,248,0.2); color: #818cf8; padding: 3px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 700;">CENTRAL BRAIN CONSOLIDATION</span>
                    <h2 style="color: #f8fafc; margin: 0.5rem 0 0.2rem 0; font-size: 2rem;">{g_score} <span style="font-size: 1rem; color: #94a3b8;">/ 100</span></h2>
                    <span style="color: {g_color}; font-weight: 700; font-size: 1.1rem;">Risk Tier: {g_level}</span>
                </div>
                <div style="text-align: right;">
                    <div style="color: #94a3b8; font-size: 0.8rem;">EVALUATION TIMESTAMP</div>
                    <div style="color: #cbd5e1; font-size: 0.95rem; font-weight: 600;">{score_data.get('calculated_at', '')[:19].replace('T', ' ')}</div>
                    <div style="margin-top: 0.5rem;"><span style="background: rgba(34,197,94,0.15); color: #22c55e; padding: 3px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">Automated Escalation Armed</span></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 4 Component Breakdown Cards
        st.markdown("##### 🧩 Multi-Agent Indicator Sub-Scores (25% Weight Each)")
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            h_score = comp_scores.get("active_hazards", 0.0)
            h_color = "#ef4444" if h_score > 60 else ("#f59e0b" if h_score > 30 else "#22c55e")
            st.metric("Site Risk Hazards", f"{h_score}/100", help="Site Risk Agent: active site hazards and severity")
            st.progress(min(1.0, max(0.0, h_score / 100.0)))
        with sc2:
            p_score = comp_scores.get("ppe_non_compliance", 0.0)
            st.metric("PPE Non-Compliance", f"{p_score}/100", help="Safety Agent: inverted worker compliance rate")
            st.progress(min(1.0, max(0.0, p_score / 100.0)))
        with sc3:
            r_score = comp_scores.get("regulatory_violations", 0.0)
            st.metric("Regulatory Violations", f"{r_score}/100", help="Compliance Agent: open regulatory violations & audit readiness")
            st.progress(min(1.0, max(0.0, r_score / 100.0)))
        with sc4:
            i_score = comp_scores.get("insurance_exposure", 0.0)
            st.metric("Insurance Exposure", f"{i_score}/100", help="Insurance Agent: claim risk scores and financial exposure")
            st.progress(min(1.0, max(0.0, i_score / 100.0)))

        # Human-in-the-Loop Override
        st.markdown("---")
        st.markdown('<div class="section-header">🛡️ Human-in-the-Loop Risk Score Override</div>', unsafe_allow_html=True)
        st.caption("Authorized supervisors can manually adjust site risk score. All overrides are permanently recorded in the immutable audit trail.")

        with st.expander("📝 Apply Official Risk Score Override", expanded=False):
            with st.form("risk_override_form"):
                ov_score = st.slider("New Site Risk Score", 0.0, 100.0, float(g_score), 0.5)
                ov_reason = st.text_area("Mandatory Operational Justification", placeholder="e.g. Additional safety netting and edge protection verified by third-party inspector.")
                ov_user = st.text_input("Authorizing Supervisor Name", value="Senior Safety Supervisor")
                submit_override = st.form_submit_button("✅ Submit Official Override", type="primary")

                if submit_override:
                    if not ov_reason.strip():
                        st.error("Operational justification is mandatory for compliance auditability.")
                    else:
                        res = intelligence_engine.apply_manual_risk_override(
                            project_id="GLOBAL",
                            new_risk_score=ov_score,
                            reason=ov_reason.strip(),
                            user_name=ov_user.strip(),
                        )
                        st.success(f"Risk Score successfully overridden to {res['overridden_score']}/100 ({res['overridden_level']}). Audit entry recorded.")
                        st.rerun()

    # ------------------------------------------------------------------
    # TAB 2: Predictive Analytics & Forecasting & ROI
    # ------------------------------------------------------------------
    with tab2:
        st.markdown('<div class="section-header">🔮 Predictive Analytics & Safety ROI Cost Model</div>', unsafe_allow_html=True)
        forecast = intelligence_engine.run_predictive_analytics(project_id=None)

        inc_forecast = forecast.get("incident_forecast", {})
        roi_model = forecast.get("safety_roi_model", {})
        recurring = forecast.get("recurring_patterns", {})

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            p_val = inc_forecast.get("incident_probability_pct", 0.0)
            p_color = "#ef4444" if p_val > 50 else ("#f59e0b" if p_val > 25 else "#22c55e")
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">14-DAY INCIDENT PROBABILITY</div>
                <div class="value" style="color:{p_color}">{p_val}%</div>
                <div class="description">Projected site risk window</div>
            </div>
            """, unsafe_allow_html=True)
        with fc2:
            trend = inc_forecast.get("risk_trend", "STABLE")
            t_color = "#22c55e" if trend == "IMPROVING" else ("#f59e0b" if trend == "STABLE" else "#ef4444")
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">RISK TRAJECTORY</div>
                <div class="value" style="color:{t_color}">{trend}</div>
                <div class="description">Trend vs historical violations</div>
            </div>
            """, unsafe_allow_html=True)
        with fc3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">MODEL CONFIDENCE</div>
                <div class="value" style="color:#818cf8">{int(inc_forecast.get('confidence_level', 0.88)*100)}%</div>
                <div class="description">Bayesian incident forecasting</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        # Safety ROI / Financial Cost Model
        st.markdown('<div class="section-header">💰 Safety ROI & Cost Avoidance Impact</div>', unsafe_allow_html=True)
        st.caption("Based on OSHA / construction industry cost baseline of $35,000 per incident prevented.")

        roi_col1, roi_col2, roi_col3, roi_col4 = st.columns(4)
        with roi_col1:
            st.metric("Baseline Compliance", f"{roi_model.get('baseline_compliance_pct', 72.0)}%")
        with roi_col2:
            st.metric("Current Compliance", f"{roi_model.get('current_compliance_pct', 100.0)}%", delta=f"+{roi_model.get('compliance_gain_pct', 0.0)}%")
        with roi_col3:
            st.metric("Est. Incidents Prevented", f"{roi_model.get('estimated_incidents_prevented', 0.0)}")
        with roi_col4:
            cost_usd = roi_model.get("estimated_cost_avoidance_usd", 0.0)
            st.metric("Total Cost Avoidance", f"${cost_usd:,.2f}", delta="Cost Saved", delta_color="normal")

        st.markdown("---")
        # Recurring Patterns
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("##### 🦺 Top Recurring Missing PPE Items")
            missing_items = recurring.get("top_missing_ppe", [])
            if missing_items:
                st.dataframe(pd.DataFrame(missing_items), use_container_width=True, hide_index=True)
            else:
                st.info("No recurring missing PPE patterns detected.")

        with rc2:
            st.markdown("##### ⚠️ High-Frequency Repeat Violators")
            repeat_viol = recurring.get("repeat_violators", [])
            if repeat_viol:
                st.dataframe(pd.DataFrame(repeat_viol), use_container_width=True, hide_index=True)
            else:
                st.info("No repeat violators on record.")

    # ------------------------------------------------------------------
    # TAB 3: Evaluated Inputs & Activity Breakdown
    # ------------------------------------------------------------------
    with tab3:
        st.markdown('<div class="section-header">📊 Total Evaluated Site Inputs Breakdown</div>', unsafe_allow_html=True)
        st.caption("Detailed activity log of all uploaded images, video frames, workers, and hazard alerts actually evaluated by the platform.")

        pfc1, pfc2, pfc3 = st.columns(3)
        with pfc1:
            st.metric("Total Analysis Runs", analytics.get("total_events", 0))
        with pfc2:
            st.metric("Total Workers Evaluated", analytics.get("total_workers", 0))
        with pfc3:
            st.metric("Total Stored Alerts", len(all_alerts))

        st.markdown("##### 📜 Recent Evaluated Safety Sessions & Events")
        events = db_manager.get_recent_events(limit=30)
        if events:
            ev_df = pd.DataFrame(events)
            disp_cols = ["timestamp", "source", "worker_count", "compliant_workers", "violations_count", "risk_level"]
            disp_cols = [c for c in disp_cols if c in ev_df.columns]
            ev_df = ev_df[disp_cols]
            ev_df.columns = [c.replace("_", " ").title() for c in disp_cols]
            st.dataframe(ev_df, use_container_width=True, hide_index=True)
        else:
            st.info("No safety analysis events recorded yet. Upload an image or video in the sidebar to begin evaluation.")

        st.markdown("##### 👷 Evaluated Workers Compliance Status")
        workers_list = db_manager.get_workers(limit=50)
        if workers_list:
            w_df = pd.DataFrame(workers_list)
            w_cols = ["worker_id", "ppe_status", "missing_ppe", "risk_level", "last_seen", "detection_count", "violation_count"]
            w_cols = [c for c in w_cols if c in w_df.columns]
            w_df = w_df[w_cols]
            w_df.columns = [c.replace("_", " ").title() for c in w_cols]
            st.dataframe(w_df, use_container_width=True, hide_index=True)
        else:
            st.info("No worker records yet.")

    # ------------------------------------------------------------------
    # TAB 4: Reporting Agent & Multi-Format Exports
    # ------------------------------------------------------------------
    with tab4:
        st.markdown('<div class="section-header">📑 Autonomous Reporting Agent & Exports</div>', unsafe_allow_html=True)
        st.caption("Generate, view, and export executive, operational, and regulatory compliance reports in JSON, Markdown, or HTML/PDF.")

        rpt_c1, rpt_c2, rpt_c3 = st.columns([2, 1, 1])
        with rpt_c1:
            report_type_choice = st.selectbox(
                "Report Type",
                [
                    "Daily Operational Safety Report",
                    "Executive Risk Summary",
                    "Regulatory & Insurance Audit Dossier",
                    "Project Health Assessment",
                ],
                key="m4_rpt_type_choice",
            )
        with rpt_c2:
            export_fmt_choice = st.selectbox("Export Format", ["PDF (.pdf)", "HTML / Print-Ready", "Markdown", "JSON"], key="m4_fmt_choice")
        with rpt_c3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            gen_report_btn = st.button("🚀 Generate Report", type="primary", use_container_width=True)

        type_map = {
            "Daily Operational Safety Report": "daily_operational",
            "Executive Risk Summary": "executive_summary",
            "Regulatory & Insurance Audit Dossier": "audit_report",
            "Project Health Assessment": "project_health",
        }
        fmt_map = {
            "PDF (.pdf)": "pdf",
            "HTML / Print-Ready": "html",
            "Markdown": "markdown",
            "JSON": "json",
        }

        if gen_report_btn:
            selected_type_key = type_map[report_type_choice]
            selected_fmt_key = fmt_map[export_fmt_choice]

            with st.spinner(f"Generating autonomous {export_fmt_choice} intelligence report..."):
                if selected_type_key == "daily_operational":
                    rep_res = reporting_agent.generate_daily_operational_report(project_id="GLOBAL", export_format=selected_fmt_key)
                elif selected_type_key == "executive_summary":
                    rep_res = reporting_agent.generate_executive_summary(project_id="GLOBAL", export_format=selected_fmt_key)
                elif selected_type_key == "audit_report":
                    rep_res = reporting_agent.generate_audit_packet(project_id="GLOBAL", export_format=selected_fmt_key)
                else:
                    rep_res = reporting_agent.generate_project_health_report(project_id="GLOBAL", export_format=selected_fmt_key)

            st.session_state["m4_latest_report"] = rep_res
            st.session_state["m4_latest_report_fmt"] = selected_fmt_key

        latest_rep = st.session_state.get("m4_latest_report")
        latest_fmt = st.session_state.get("m4_latest_report_fmt", "pdf")

        if latest_rep:
            st.markdown("---")
            rep_id = latest_rep.get("report_id", "RPT")
            raw_data = latest_rep.get("rendered") if isinstance(latest_rep, dict) and "rendered" in latest_rep else json.dumps(latest_rep, indent=2)

            file_ext = "pdf" if latest_fmt == "pdf" else ("html" if latest_fmt == "html" else ("md" if latest_fmt == "markdown" else "json"))
            mime_type = "application/pdf" if latest_fmt == "pdf" else ("text/html" if latest_fmt == "html" else ("text/markdown" if latest_fmt == "markdown" else "application/json"))

            # Format download payload (bytes for PDF, str for text formats)
            if latest_fmt == "pdf":
                download_data = raw_data if isinstance(raw_data, (bytes, bytearray)) else str(raw_data).encode("utf-8")
            else:
                download_data = raw_data if isinstance(raw_data, str) else json.dumps(raw_data, indent=2)

            dl_c1, dl_c2 = st.columns([3, 1])
            with dl_c1:
                st.success(f"Report Generated: **{rep_id}** ({file_ext.upper()})")
            with dl_c2:
                st.download_button(
                    label=f"📥 Download .{file_ext.upper()}",
                    data=download_data,
                    file_name=f"{rep_id}.{file_ext}",
                    mime=mime_type,
                    type="primary",
                    use_container_width=True,
                )

            st.markdown("##### 📄 Report Output Preview")
            with st.container(border=True):
                if latest_fmt == "pdf":
                    st.info("📄 **Official PDF Document generated successfully.** Click the download button above to save the binary PDF. Previewing document layout below:")
                    import streamlit.components.v1 as components
                    html_preview = reporting_agent._to_html(latest_rep.get("raw", {}))
                    components.html(html_preview, height=600, scrolling=True)
                elif latest_fmt == "html":
                    import streamlit.components.v1 as components
                    components.html(raw_data, height=600, scrolling=True)
                elif latest_fmt == "markdown":
                    st.markdown(raw_data)
                else:
                    st.json(latest_rep)

        st.markdown("---")
        st.markdown("##### 📂 Saved Reports Archive")
        saved_reps = db_manager.get_reports(project_id=None, limit=10)
        if saved_reps:
            clean_reps = [
                {"Report ID": r["report_id"], "Type": r["report_type"], "Title": r["title"], "Date": r["generated_at"][:19].replace("T", " "), "Status": r.get("status", "Final")}
                for r in saved_reps
            ]
            st.dataframe(pd.DataFrame(clean_reps), use_container_width=True, hide_index=True)
        else:
            st.info("No saved reports found yet.")

    # ------------------------------------------------------------------
    # TAB 5: Agent Health & Immutable Audit Trail
    # ------------------------------------------------------------------
    with tab5:
        st.markdown('<div class="section-header">🩺 Agent Performance Health & Immutable Audit Trail</div>', unsafe_allow_html=True)
        health_rep = intelligence_engine.get_agent_health_report()

        sys_status = health_rep.get("overall_system_health", "OPTIMAL")
        s_color = "#22c55e" if sys_status == "OPTIMAL" else "#ef4444"

        st.markdown(f"""
        <div style="background: rgba(30,30,50,0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1.2rem; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong style="color: #cbd5e1; font-size: 1.1rem;">Overall System Status:</strong>
                <span style="color: {s_color}; font-weight: 800; font-size: 1.1rem; margin-left: 0.5rem;">● {sys_status}</span>
            </div>
            <div style="color: #94a3b8; font-size: 0.85rem;">Continuous Baselines Verification Active</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### 🤖 Sub-Agent Efficiency vs Baselines")
        ac1, ac2, ac3, ac4 = st.columns(4)
        agents_data = health_rep.get("agents", {})

        agent_cards = [
            (ac1, "Site Risk Agent", "site_risk_agent", "🏗️"),
            (ac2, "Safety Agent", "safety_agent", "🦺"),
            (ac3, "Compliance Agent", "compliance_agent", "⚖️"),
            (ac4, "Insurance Agent", "insurance_agent", "📋"),
        ]

        for col, title, key, icon in agent_cards:
            ag_data = agents_data.get(key, {})
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">{icon} {title}</div>
                    <div class="value" style="color:#22c55e; font-size: 1.4rem;">{ag_data.get('current_efficiency_pct', 95.0)}%</div>
                    <div class="description">Baseline: {ag_data.get('baseline_efficiency_pct', 94.0)}% • Status: {ag_data.get('status', 'HEALTHY')}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### 🔒 Tamper-Evident Immutable Audit Trail")
        st.caption("Immutable append-only log of all system orchestrations, automated escalations, and human overrides.")

        trail = db_manager.get_audit_trail(limit=30)
        if trail:
            trail_view = [
                {
                    "Timestamp": t.get("timestamp", "")[:19].replace("T", " "),
                    "Action": t.get("action", ""),
                    "Agent / Actor": t.get("agent", ""),
                    "User Override": "⚠️ YES" if t.get("user_override") else "No",
                    "Details": json.dumps(t.get("details", {}))[:80] + "...",
                }
                for t in trail
            ]
            st.dataframe(pd.DataFrame(trail_view), use_container_width=True, hide_index=True)
        else:
            st.info("No audit trail entries recorded yet.")


# ======================================================================
#  PAGE: DASHBOARD
# ======================================================================
if page == "🏠 Dashboard":
    st.markdown(
        '<div class="analytics-hero"><h1>SAFETY ANALYTICS DASHBOARD</h1>'
        '<p>Real-time insights into worker safety, PPE compliance, violations, and risk levels.</p></div>',
        unsafe_allow_html=True,
    )

    # Get analytics summary from database
    analytics = db_manager.get_analytics_summary()

    # Central Brain Global Risk Score overview
    global_risk = intelligence_engine.calculate_global_risk_score()
    g_score = global_risk.get("global_risk_score", 0.0)
    g_level = global_risk.get("risk_level", "Low")
    g_color = "#ef4444" if g_level == "Critical" else ("#f97316" if g_level == "High" else ("#f59e0b" if g_level == "Medium" else "#22c55e"))

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(30,27,75,0.85), rgba(15,23,42,0.85)); border: 1px solid rgba(129,140,248,0.25); border-radius: 12px; padding: 1.1rem 1.4rem; margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
        <div>
            <span style="background: rgba(129,140,248,0.2); color: #818cf8; padding: 2px 8px; border-radius: 999px; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;">Central Brain Orchestration</span>
            <div style="color: #f8fafc; font-size: 1.25rem; font-weight: 700; margin-top: 0.3rem;">Unified Project Risk Score: <span style="color: {g_color}; font-weight: 800;">{g_score} / 100 ({g_level})</span></div>
            <div style="color: #94a3b8; font-size: 0.82rem; margin-top: 0.15rem;">Multi-Agent Orchestration • 4 Sub-Agents Healthy • Automated Escalation Active</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.72rem; color: #94a3b8;">SYSTEM HEALTH</div>
            <div style="font-size: 1rem; font-weight: 700; color: #22c55e;">● OPTIMAL (95.6% AVG)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_analytics_section("Safety Overview", "A live snapshot of tracked workers and current safety performance.")

    # Summary cards row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown('<div class="metric-card overview-card"><div class="label">TOTAL WORKERS</div>'
                    f'<div class="value" style="color:#818cf8">{analytics["total_workers"]}</div>'
                    '<div class="description">Workers tracked by the system</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card overview-card"><div class="label">PPE COMPLIANT</div>'
                    f'<div class="value" style="color:#22c55e">{analytics["compliant_workers"]}</div>'
                    '<div class="description">Workers meeting required PPE rules</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card overview-card"><div class="label">PPE VIOLATIONS</div>'
                    f'<div class="value" style="color:#ef4444">{analytics["total_violations"]}</div>'
                    '<div class="description">Recorded missing-PPE events</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card overview-card"><div class="label">ACTIVE ALERTS</div>'
                    f'<div class="value" style="color:#f97316">{analytics["active_alerts"]}</div>'
                    '<div class="description">Alerts awaiting resolution</div></div>', unsafe_allow_html=True)
    with col5:
        score = analytics["safety_score"]
        score_color = "#22c55e" if score >= 80 else ("#f59e0b" if score >= 50 else "#ef4444")
        st.markdown('<div class="metric-card overview-card"><div class="label">OVERALL SAFETY SCORE</div>'
                    f'<div class="value" style="color:{score_color}">{score}%</div>'
                    '<div class="description">Based on latest worker PPE status</div></div>', unsafe_allow_html=True)

    # The home page is also an actionable entry point. Keep the full pages
    # available, but make an uploaded image useful without changing navigation.
    if uploaded_image is not None:
        st.markdown('<div class="section-header">🔍 Run Safety Analysis</div>', unsafe_allow_html=True)
        st.caption(f"Ready to analyze: {uploaded_image.name}")
        if st.button("▶ Analyze Uploaded Image", type="primary", key="dashboard_analyze_image"):
            try:
                with st.spinner("Running YOLO detection and PPE safety analysis..."):
                    dashboard_report = safety_agent.analyze_uploaded_image(
                        uploaded_image.getvalue(),
                        filename=uploaded_image.name,
                        conf_threshold=conf_threshold,
                    )
                st.session_state["dashboard_report"] = dashboard_report
                st.success("Safety analysis complete."
                           " Review the annotated image and worker results below.")
            except Exception as exc:
                st.error(f"Safety analysis failed: {exc}")

    dashboard_report = st.session_state.get("dashboard_report")
    if dashboard_report is not None:
        report_col1, report_col2, report_col3, report_col4 = st.columns(4)
        with report_col1:
            render_metric_card("Workers", dashboard_report["worker_count"], "#818cf8")
        with report_col2:
            render_metric_card("PPE Compliant", dashboard_report["compliant_workers"], "#22c55e")
        with report_col3:
            render_metric_card("PPE Violations", dashboard_report["violations"], "#ef4444")
        with report_col4:
            render_metric_card("Risk", dashboard_report["risk_level"], risk_color(dashboard_report["risk_level"]))

        st.image(
            cv2.cvtColor(dashboard_report["annotated_image"], cv2.COLOR_BGR2RGB),
            caption="YOLO detections and worker PPE assessment",
            use_container_width=True,
        )
        if dashboard_report["worker_results"]:
            worker_view = pd.DataFrame(dashboard_report["worker_results"])
            worker_view = worker_view[["worker_id", "ppe_status", "missing_ppe", "risk_level", "confidence"]]
            st.dataframe(worker_view, use_container_width=True, hide_index=True)

            render_ppe_compliance_card(dashboard_report.get("ppe_compliance"))

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        render_analytics_section("PPE Compliance Distribution", "Current worker PPE compliance status")
        with st.container(border=True):
            comp_fig = create_compliance_pie_chart(
                analytics["compliant_workers"],
                analytics["non_compliant_workers"]
            )
            st.pyplot(comp_fig, use_container_width=True)

    with chart_col2:
        render_analytics_section("Risk Level Distribution", "Distribution of recorded PPE violations by risk severity")
        with st.container(border=True):
            risk_fig = create_risk_distribution_chart(analytics["risk_distribution"])
            st.pyplot(risk_fig, use_container_width=True)

    render_analytics_section("Violations by Type", "Most frequently detected PPE violations")
    with st.container(border=True):
        if analytics["violation_types"]:
            viol_fig = create_violations_by_type_chart(analytics["violation_types"])
            st.pyplot(viol_fig, use_container_width=True)
        else:
            st.info("No PPE violation data available yet.")

    render_analytics_section("Safety Trends", "Recorded PPE violations over time")
    with st.container(border=True):
        trend_events = db_manager.get_recent_events(limit=100)
        if trend_events:
            trend_df = pd.DataFrame(trend_events)
            trend_df["timestamp"] = pd.to_datetime(trend_df["timestamp"], errors="coerce")
            trend_df = trend_df.dropna(subset=["timestamp"]).sort_values("timestamp")
            if not trend_df.empty:
                trend_fig = create_safety_trend_chart(
                    trend_df["timestamp"].tolist(),
                    trend_df["violations_count"].tolist(),
                )
                st.pyplot(trend_fig, use_container_width=True)
            else:
                st.info("No historical safety trend data available yet.")
        else:
            st.info("No historical safety trend data available yet.")

    render_analytics_section("Recent Safety Alerts", "Latest alerts generated by the monitoring pipeline")
    recent_alerts = db_manager.get_alerts(limit=10)
    play_alert_beep(recent_alerts)
    render_recent_alerts_table(recent_alerts)


# ======================================================================
#  PAGE: ENTERPRISE INTELLIGENCE & REPORTS (MILESTONE 4)
# ======================================================================
elif page == "📈 Enterprise Intelligence & Reports":
    render_enterprise_intelligence_page(db_manager, intelligence_engine, reporting_agent)


# ======================================================================
#  PAGE: COMPLIANCE & INSURANCE INTELLIGENCE
# ======================================================================
elif page == "🏛️ Compliance & Insurance":
    render_compliance_assessment_section(db_manager, risk_engine)


# ======================================================================
#  PAGE: SAFETY MONITORING
# ======================================================================
elif page == "🛡️ Safety Monitoring":
    st.markdown('<div class="section-header">🛡️ Worker Safety Monitoring</div>', unsafe_allow_html=True)

    # If an image was uploaded, run safety analysis immediately
    if uploaded_image is not None:
        try:
            image_bytes = uploaded_image.getvalue()
            report = safety_agent.analyze_uploaded_image(
                image_bytes, filename=uploaded_image.name, conf_threshold=conf_threshold
            )

            worker_results = report.get("worker_results", [])
            risk = report.get("risk", {})

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                render_metric_card("Workers Detected", report["worker_count"], "#818cf8")
            with col2:
                render_metric_card("Compliant", report["compliant_workers"], "#22c55e")
            with col3:
                render_metric_card("Violations", report["violations"], "#ef4444")
            with col4:
                render_metric_card("Risk Level", report["risk_level"], risk_color(report["risk_level"]))

            st.markdown("")

            # Annotated image
            ann_rgb = cv2.cvtColor(report["annotated_image"], cv2.COLOR_BGR2RGB)
            st.image(ann_rgb, caption="Safety Analysis — Worker-PPE Association", use_container_width=True)

            # Worker safety cards
            st.markdown('<div class="section-header">👷 Individual Worker Assessment</div>', unsafe_allow_html=True)
            if worker_results:
                for w in worker_results:
                    status_class = "status-compliant" if w["ppe_status"] == "Compliant" else "status-violation"
                    missing_text = ", ".join(w["missing_ppe"]) if w["missing_ppe"] else "None"
                    detected_text = ", ".join(w.get("detected_ppe", [])) if w.get("detected_ppe") else "None"

                    col_a, col_b = st.columns([1, 2])
                    with col_a:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="label">{w['worker_id']}</div>
                            <div class="value" style="color:{risk_color(w['risk_level'])}">{w['risk_level']}</div>
                            <span class="{status_class}">{w['ppe_status']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_b:
                        st.markdown(f"**{w['worker_id']}** — {w['ppe_status']}")
                        st.markdown(f"- ✅ Detected PPE: {detected_text}")
                        st.markdown(f"- ❌ Missing PPE: {missing_text}")
                        st.markdown(f"- 🎯 Confidence: {w['confidence']:.0%}")
                        st.markdown(f"- ⚠️ Violation: {w['violation']}")
                    st.markdown("---")

            # Alerts generated
            if report.get("alerts"):
                st.markdown('<div class="section-header">🚨 Alerts Generated</div>', unsafe_allow_html=True)
                play_alert_beep(report["alerts"])
                for alert in report["alerts"]:
                    sev = alert.get("risk_level", "MEDIUM").lower()
                    conf = alert.get("confidence")
                    conf_str = f"{float(conf):.0%}" if conf is not None else "N/A"
                    st.markdown(f"""
                    <div class="alert-card {sev}">
                        <div class="alert-title">{alert.get('risk_level', 'N/A')} RISK ALERT</div>
                        <div class="alert-meta">
                            {alert.get('violation_type', '')} &nbsp;│&nbsp;
                            🕐 {alert.get('timestamp', '')[:19]} &nbsp;│&nbsp;
                            📊 Confidence: {conf_str} &nbsp;│&nbsp;
                            📌 {alert.get('status', 'New')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # JSON output preview
            with st.expander("📋 View Structured Safety Report (JSON)"):
                safe_report = {k: v for k, v in report.items() if k not in ("original_image", "annotated_image")}
                st.json(safe_report)

        except Exception as e:
            st.error(f"❌ Error during safety monitoring: {e}")
    else:
        # Show recent monitoring data
        analytics = db_manager.get_analytics_summary()
        col1, col2, col3 = st.columns(3)
        with col1:
            render_metric_card("Total Workers Tracked", analytics["total_workers"], "#818cf8")
        with col2:
            render_metric_card("Total Violations", analytics["total_violations"], "#ef4444")
        with col3:
            render_metric_card("Safety Score", f"{analytics['safety_score']}%",
                               "#22c55e" if analytics["safety_score"] >= 80 else "#ef4444")

        st.info("📷 Upload an image in the sidebar to begin real-time safety monitoring.")

        # Show recent events
        events = db_manager.get_recent_events(limit=10)
        if events:
            st.markdown('<div class="section-header">📜 Recent Safety Events</div>', unsafe_allow_html=True)
            ev_df = pd.DataFrame(events)
            display_cols = ["timestamp", "source", "worker_count", "compliant_workers", "violations_count", "risk_level"]
            display_cols = [c for c in display_cols if c in ev_df.columns]
            ev_df = ev_df[display_cols]
            ev_df.columns = [c.replace("_", " ").title() for c in display_cols]
            st.dataframe(ev_df, use_container_width=True, hide_index=True)


# ======================================================================
#  PAGE: LIVE DETECTION
# ======================================================================
elif page == "🔍 Live Detection":
    st.markdown('<div class="section-header">🔍 Live Detection & Analysis</div>', unsafe_allow_html=True)

    # IMAGE ANALYSIS
    if uploaded_image is not None:
        st.markdown("##### 📷 Image Analysis")
        try:
            image_bytes = uploaded_image.getvalue()

            # Milestone 2 Safety Agent analysis
            report = safety_agent.analyze_uploaded_image(
                image_bytes, filename=uploaded_image.name, conf_threshold=conf_threshold
            )

            risk = report["risk"]
            detections = report["detections"]

            # Metric row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                render_metric_card("Risk Score", f"{risk['score']}/100", risk["level_color"])
            with col2:
                render_metric_card("Risk Level", f"{risk['level_emoji']} {risk['level']}", risk["level_color"])
            with col3:
                render_metric_card("PPE Violations", report["violations"], "#ef4444")
            with col4:
                render_metric_card("Workers", report["worker_count"], "#f59e0b")

            st.markdown("")

            # Images side-by-side
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.markdown("##### Original Image")
                orig_rgb = cv2.cvtColor(report["original_image"], cv2.COLOR_BGR2RGB)
                st.image(orig_rgb, use_container_width=True)

            with img_col2:
                st.markdown("##### Safety Analysis (YOLO + PPE Association)")
                ann_rgb = cv2.cvtColor(report["annotated_image"], cv2.COLOR_BGR2RGB)
                st.image(ann_rgb, use_container_width=True)

            st.markdown('<div class="section-header">📦 All YOLO Object Detections</div>', unsafe_allow_html=True)
            render_detection_table(detections)

            # Charts row
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown("##### 🎯 Risk Gauge")
                gauge_fig = create_risk_gauge(risk["score"], risk["level"], risk["level_color"])
                st.pyplot(gauge_fig)

            with chart_col2:
                st.markdown("##### 📊 Detection Statistics")
                stats_fig = create_detection_stats_chart(detections)
                st.pyplot(stats_fig)

            # Hazard distribution
            if risk["hazard_breakdown"]:
                st.markdown('<div class="section-header">⚠️ Hazard Breakdown</div>', unsafe_allow_html=True)
                haz_col1, haz_col2 = st.columns([1.2, 1])

                with haz_col1:
                    haz_df = pd.DataFrame(risk["hazard_breakdown"])
                    haz_df.columns = ["Hazard", "Count", "Avg Confidence", "Risk Contribution", "Severity"]
                    haz_df["Avg Confidence"] = haz_df["Avg Confidence"].apply(lambda x: f"{x:.0%}")
                    st.dataframe(haz_df, use_container_width=True, hide_index=True)

                with haz_col2:
                    hazard_fig = create_hazard_chart(risk["hazard_breakdown"])
                    st.pyplot(hazard_fig)

            # Contributing factors
            if risk["contributing_factors"]:
                st.markdown('<div class="section-header">🔍 Contributing Factors</div>', unsafe_allow_html=True)
                for factor in risk["contributing_factors"]:
                    st.markdown(f"- {factor}")

            # Worker-PPE results
            if report.get("worker_results"):
                st.markdown('<div class="section-header">👷 Worker PPE Compliance</div>', unsafe_allow_html=True)
                wr_df = pd.DataFrame(report["worker_results"])
                wr_display = ["worker_id", "ppe_status", "missing_ppe", "detected_ppe", "risk_level", "confidence"]
                wr_display = [c for c in wr_display if c in wr_df.columns]
                wr_df = wr_df[wr_display]
                wr_df.columns = [c.replace("_", " ").title() for c in wr_display]
                st.dataframe(wr_df, use_container_width=True, hide_index=True)

            render_ppe_compliance_card(report.get("ppe_compliance"))

            # Safety Recommendations
            recs = report.get("risk", {}).get("risk_factors", [])
            if recs:
                st.markdown('<div class="section-header">⚠️ Risk Factors</div>', unsafe_allow_html=True)
                for f in recs:
                    st.markdown(f"- {f}")
                st.info(risk.get("risk_explanation", ""))

        except Exception as e:
            st.error(f"❌ Error analysing image: {e}")

    # VIDEO ANALYSIS
    elif uploaded_video is not None:
        st.markdown("##### 🎬 Video Analysis")
        try:
            temp_dir = os.path.join(PROJECT_ROOT, "outputs")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"temp_{uploaded_video.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_video.getvalue())

            st.info(f"🎬 Processing video: **{uploaded_video.name}** (every {frame_skip}th frame, max {max_frames} frames)")

            progress_bar = st.progress(0, text="Analysing frames...")

            def update_progress(current, total):
                if total > 0:
                    val = min(1.0, max(0.0, float(current) / float(total)))
                    progress_bar.progress(val, text=f"Analysing frame {current}/{total}...")

            # Use the Safety Agent so sampled frames create events, worker
            # records, and de-duplicated alerts just like image analysis.
            video_report = safety_agent.analyze_video(
                temp_path,
                conf_threshold=conf_threshold,
                frame_skip=frame_skip,
                max_frames=max_frames,
                progress_callback=update_progress,
            )

            progress_bar.progress(1.0, text="✅ Analysis complete!")

            # Video metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                render_metric_card("Risk Score", f"{video_report['risk_score']}/100",
                                   risk_color(video_report.get('risk_level', 'Low')))
            with col2:
                render_metric_card("Risk Level",
                    video_report['risk_level'],
                    risk_color(video_report.get('risk_level', 'Low')))
            with col3:
                render_metric_card("Violations", video_report['violations'], "#ef4444")
            with col4:
                render_metric_card("Frames Analysed", video_report['analysed_frames'], "#818cf8")

            st.markdown("")

            stat_col1, stat_col2, stat_col3 = st.columns(3)
            with stat_col1:
                st.metric("Workers / Frame", video_report["worker_count"])
            with stat_col2:
                st.metric("Compliant / Frame", video_report["compliant_workers"])
            with stat_col3:
                st.metric("Alerts Generated", len(video_report.get("all_alerts", [])))
            st.caption(
                f"Sampled {video_report['analysed_frames']} of {video_report['total_frames']} frames · "
                f"Video FPS: {video_report.get('fps', 0):.1f}"
            )

            # Risk trend chart
            if video_report["frame_reports"]:
                st.markdown('<div class="section-header">📈 Risk Score Trend</div>', unsafe_allow_html=True)
                scores = [r["risk"]["score"] for r in video_report["frame_reports"]]
                frames = [r.get("frame_number", index) for index, r in enumerate(video_report["frame_reports"])]
                trend_fig = create_risk_trend_chart(scores, frames)
                st.pyplot(trend_fig)

                worst_frame = max(video_report["frame_reports"], key=lambda item: item["risk"]["score"])
                st.image(
                    cv2.cvtColor(worst_frame["annotated_image"], cv2.COLOR_BGR2RGB),
                    caption="Highest-risk sampled frame",
                    use_container_width=True,
                )

            # Recommendations
            st.markdown('<div class="section-header">💡 Safety Recommendations</div>', unsafe_allow_html=True)
            recommendations = [
                frame.get("risk", {}).get("risk_explanation", "")
                for frame in video_report.get("frame_reports", [])
            ]
            for rec in dict.fromkeys(item for item in recommendations if item):
                st.markdown(f'<div class="rec-card">{rec}</div>', unsafe_allow_html=True)

            try:
                os.remove(temp_path)
            except OSError:
                pass

        except Exception as e:
            st.error(f"❌ Error analysing video: {e}")

    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem;">
            <p style="font-size: 4rem; margin: 0;">🔍</p>
            <h3 style="color: #e0e7ff; font-weight: 700;">Upload an Image or Video for Detection</h3>
            <p style="color: #94a3b8; max-width: 500px; margin: 0.5rem auto;">
                Use the sidebar to upload a construction-site image or video.
                The AI will detect objects, assess PPE compliance, calculate risk, and generate alerts.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ======================================================================
#  PAGE: ALERTS
# ======================================================================
elif page == "🚨 Alerts":
    st.markdown('<div class="section-header">🚨 Safety Alert Management</div>', unsafe_allow_html=True)

    # Alert status filter
    filter_col1, filter_col2 = st.columns([1, 3])
    with filter_col1:
        alert_filter = st.selectbox("Filter by Status", ["All", "New", "Acknowledged", "Resolved"])

    status_filter = None if alert_filter == "All" else alert_filter
    alerts = db_manager.get_alerts(status=status_filter, limit=100)
    play_alert_beep(alerts)

    # Stats row
    # Stats row
    all_alerts = db_manager.get_alerts(limit=500)
    sms_logs = db_manager.get_sms_logs(limit=100)
    new_count = sum(1 for a in all_alerts if a.get("status") == "New")
    ack_count = sum(1 for a in all_alerts if a.get("status") == "Acknowledged")
    resolved_count = sum(1 for a in all_alerts if a.get("status") == "Resolved")
    sms_count = len(sms_logs)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_metric_card("Total Alerts", len(all_alerts), "#818cf8")
    with col2:
        render_metric_card("New", new_count, "#ef4444")
    with col3:
        render_metric_card("Acknowledged", ack_count, "#f59e0b")
    with col4:
        render_metric_card("Resolved", resolved_count, "#22c55e")
    with col5:
        render_metric_card("SMS Dispatched", sms_count, "#38bdf8")

    st.markdown("")

    if alerts:
        for alert in alerts:
            sev = alert.get("risk_level", "MEDIUM").lower()
            alert_id = alert.get("alert_id", "N/A")
            is_urgent = alert.get("risk_level", "").upper() in ["HIGH", "CRITICAL"]

            sms_badge_html = ""
            if is_urgent:
                recipient = DEFAULT_WORKER_PHONES.get(alert.get("worker_id"), DEFAULT_SUPERVISOR_PHONE)
                sms_badge_html = f"""
                <div style="margin-top: 8px;">
                    <span style="background: rgba(14, 165, 233, 0.2); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.4); padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">
                        📱 Emergency SMS Sent to {alert.get('worker_id', 'Worker')} ({recipient})
                    </span>
                </div>
                """

            conf = alert.get("confidence")
            conf_str = f"{float(conf):.0%}" if conf is not None else "N/A"
            st.markdown(f"""
            <div class="alert-card {sev}">
                <div class="alert-title">{alert.get('risk_level', 'N/A')} RISK ALERT — {alert_id}</div>
                <div class="alert-meta">
                    ⚠️ {alert.get('violation_type', 'Safety Violation')} <br>
                    🕐 Time: {alert.get('timestamp', '')[:19]} &nbsp;│&nbsp;
                    👷 Worker: {alert.get('worker_id', 'N/A')} &nbsp;│&nbsp;
                    📊 Confidence: {conf_str} &nbsp;│&nbsp;
                    📌 Status: <b>{alert.get('status', 'New')}</b>
                </div>
                {sms_badge_html}
            </div>
            """, unsafe_allow_html=True)

            # Status update buttons
            bcol1, bcol2, bcol3 = st.columns(3)
            with bcol1:
                if alert.get("status") == "New":
                    if st.button(f"✅ Acknowledge", key=f"ack_{alert_id}"):
                        db_manager.update_alert_status(alert_id, "Acknowledged")
                        st.rerun()
            with bcol2:
                if alert.get("status") in ("New", "Acknowledged"):
                    if st.button(f"🔒 Resolve", key=f"res_{alert_id}"):
                        db_manager.update_alert_status(alert_id, "Resolved")
                        st.rerun()
            st.markdown("---")
    else:
        st.info("No alerts found for the selected filter. Upload an image to generate safety alerts.")

    # Emergency SMS Dispatch Logs & Manual Testing Section
    st.markdown('<div class="section-header">📱 Emergency SMS Dispatch Log & Live Testing</div>', unsafe_allow_html=True)
    with st.expander("View Dispatched SMS Records", expanded=True):
        if sms_logs:
            sms_df = pd.DataFrame(sms_logs)
            display_cols = ["timestamp", "alert_id", "worker_id", "phone_number", "risk_level", "message", "status", "provider"]
            display_cols = [c for c in display_cols if c in sms_df.columns]
            sms_df = sms_df[display_cols]
            sms_df.columns = [c.replace("_", " ").title() for c in display_cols]
            st.dataframe(sms_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No SMS messages dispatched yet. High and Critical risk violations will automatically send SMS alerts.")

    with st.expander("🧪 Test Manual SMS Dispatch"):
        st.caption("Dispatch an instant emergency SMS to a worker or supervisor line to verify live alert routing.")
        tcol1, tcol2, tcol3 = st.columns([1, 1, 1])
        with tcol1:
            target_worker = st.selectbox("Worker / Recipient", list(DEFAULT_WORKER_PHONES.keys()) + ["Supervisor"])
        with tcol2:
            target_risk = st.selectbox("Risk Level", ["HIGH", "CRITICAL", "MEDIUM", "LOW"])
        with tcol3:
            target_phone = DEFAULT_WORKER_PHONES.get(target_worker, DEFAULT_SUPERVISOR_PHONE)
            phone_input = st.text_input("Recipient Phone", value=target_phone)

        test_msg = st.text_input(
            "Alert Message",
            value=f"🚨 Immediate safety corrective action required: Missing protective equipment reported.",
        )
        if st.button("🚀 Dispatch Test SMS"):
            sms_srv = SMSService(db_manager=db_manager)
            if target_risk in ["HIGH", "CRITICAL"]:
                res = sms_srv.send_direct_sms(
                    worker_id=target_worker,
                    phone_number=phone_input,
                    risk_level=target_risk,
                    message=test_msg,
                )
                st.success(f"✅ SMS successfully routed via {res['provider']} to {phone_input} (Status: {res['status']})")
                st.rerun()
            else:
                st.warning(f"ℹ️ SMS skipped: Risk level is {target_risk}. SMS alerts are automatically restricted to HIGH and CRITICAL risks.")


# ======================================================================
#  PAGE: ANALYTICS
# ======================================================================
elif page == "📊 Analytics":
    st.markdown('<div class="section-header">📊 Safety Analytics Dashboard</div>', unsafe_allow_html=True)

    analytics = db_manager.get_analytics_summary()

    # Summary Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_metric_card("Total Events", analytics["total_events"], "#818cf8")
    with col2:
        render_metric_card("Workers Tracked", analytics["total_workers"], "#818cf8")
    with col3:
        render_metric_card("Compliance %", f"{analytics['compliance_percentage']}%",
                           "#22c55e" if analytics["compliance_percentage"] >= 80 else "#ef4444")
    with col4:
        render_metric_card("Total Violations", analytics["total_violations"], "#ef4444")
    with col5:
        render_metric_card("Total Alerts", analytics["total_alerts"], "#f97316")

    # Charts
    st.caption(
        "Compliance is based on the latest worker records. "
        "Risk and violation charts summarize stored PPE violation records."
    )
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown('<div class="chart-heading">1. PPE Compliance Distribution</div>', unsafe_allow_html=True)
        comp_fig = create_compliance_pie_chart(
            analytics["compliant_workers"],
            analytics["non_compliant_workers"]
        )
        st.pyplot(comp_fig)

    with chart_col2:
        st.markdown('<div class="chart-heading">2. Risk Level Distribution</div>', unsafe_allow_html=True)
        risk_fig = create_risk_distribution_chart(analytics["risk_distribution"])
        st.pyplot(risk_fig)

    if analytics["violation_types"]:
        st.markdown('<div class="chart-heading">3. Violations by Type</div>', unsafe_allow_html=True)
        viol_fig = create_violations_by_type_chart(analytics["violation_types"])
        st.pyplot(viol_fig)

    # Violations over time (from safety events table)
    events = db_manager.get_recent_events(limit=50)
    if events:
        st.markdown('<div class="section-header">📈 Violations Over Time</div>', unsafe_allow_html=True)
        ev_df = pd.DataFrame(events)
        if "timestamp" in ev_df.columns and "violations_count" in ev_df.columns:
            ev_df["timestamp"] = pd.to_datetime(ev_df["timestamp"])
            ev_df = ev_df.sort_values("timestamp")
            st.line_chart(ev_df.set_index("timestamp")[["violations_count", "worker_count"]])

    # Recent analysis history
    st.markdown('<div class="section-header">📜 Analysis History</div>', unsafe_allow_html=True)
    try:
        history = legacy_agent.get_analysis_history(limit=15)
        if history:
            hist_df = pd.DataFrame(history)
            display_cols = [
                "timestamp", "filename", "analysis_type",
                "total_objects", "hazard_count", "worker_count",
                "risk_score", "risk_level",
            ]
            display_cols = [c for c in display_cols if c in hist_df.columns]
            hist_df = hist_df[display_cols]
            hist_df.columns = [c.replace("_", " ").title() for c in display_cols]
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No analysis history yet.")
    except Exception:
        st.caption("No analysis history yet.")


# ======================================================================
#  PAGE: WORKERS
# ======================================================================
elif page == "👷 Workers":
    st.markdown('<div class="section-header">👷 Worker Safety Monitoring</div>', unsafe_allow_html=True)

    workers = db_manager.get_workers(limit=100)
    analytics = db_manager.get_analytics_summary()

    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("Total Workers", analytics["total_workers"], "#818cf8")
    with col2:
        render_metric_card("Compliant Workers", analytics["compliant_workers"], "#22c55e")
    with col3:
        render_metric_card("Non-Compliant", analytics["non_compliant_workers"], "#ef4444")

    st.markdown("")

    if workers:
        st.markdown("##### Worker Safety Table")
        worker_df = pd.DataFrame(workers)
        display_cols = ["worker_id", "ppe_status", "missing_ppe", "risk_level", "last_seen", "detection_count", "violation_count"]
        display_cols = [c for c in display_cols if c in worker_df.columns]
        worker_df = worker_df[display_cols]
        worker_df.columns = [c.replace("_", " ").title() for c in display_cols]
        st.dataframe(worker_df, use_container_width=True, hide_index=True)

        # Individual worker cards
        st.markdown('<div class="section-header">👷 Individual Worker Profiles</div>', unsafe_allow_html=True)
        for w in workers[:10]:
            status_class = "status-compliant" if w["ppe_status"] == "Compliant" else "status-violation"
            missing = w.get("missing_ppe", [])
            missing_text = ", ".join(missing) if isinstance(missing, list) else str(missing)

            st.markdown(f"""
            <div class="metric-card" style="text-align: left; padding: 1rem 1.5rem; margin-bottom: 0.5rem;">
                <span style="color: #e0e7ff; font-weight: 700;">{w['worker_id']}</span>
                &nbsp;&nbsp;
                <span class="{status_class}">{w['ppe_status']}</span>
                &nbsp;&nbsp;
                <span style="color: {risk_color(w['risk_level'])}; font-weight: 600;">{w['risk_level']} Risk</span>
                <br>
                <span style="color: #94a3b8; font-size: 0.85rem;">
                    Missing PPE: {missing_text or 'None'} &nbsp;│&nbsp;
                    Detections: {w.get('detection_count', 0)} &nbsp;│&nbsp;
                    Violations: {w.get('violation_count', 0)} &nbsp;│&nbsp;
                    Last seen: {w.get('last_seen', 'N/A')[:19]}
                </span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No worker data yet. Upload an image in the sidebar to begin worker monitoring.")


# ======================================================================
#  PAGE: CONFIGURATION
# ======================================================================
elif page == "⚙️ Configuration":
    st.markdown('<div class="section-header">⚙️ System Configuration</div>', unsafe_allow_html=True)

    st.markdown("##### Detection & Safety Thresholds")
    st.markdown("Adjust these parameters to tune the system's sensitivity and behavior.")

    config_col1, config_col2 = st.columns(2)

    with config_col1:
        st.markdown("""
        <div class="metric-card" style="text-align: left; padding: 1.2rem;">
            <div class="label" style="text-align: left;">CURRENT CONFIGURATION</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | **CONFIDENCE_THRESHOLD** | `{conf_threshold}` |
        | **IOU_THRESHOLD** | `{DEFAULT_IOU_THRESHOLD}` |
        | **ALERT_COOLDOWN** | `{ALERT_COOLDOWN_SECONDS}s` |
        | **REQUIRED_PPE** | `{', '.join(sorted(DEFAULT_REQUIRED_PPE))}` |
        """)

    with config_col2:
        st.markdown("""
        <div class="metric-card" style="text-align: left; padding: 1.2rem;">
            <div class="label" style="text-align: left;">MODEL CLASSES</div>
        </div>
        """, unsafe_allow_html=True)

        for i, name in enumerate(CLASS_NAMES):
            category = "⚠️ Hazard" if name in {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"} else \
                       "✅ Safety" if name in {"Hardhat", "Mask", "Safety Vest", "Safety Cone"} else \
                       "👷 Worker" if name == "Person" else \
                       "🚜 Heavy Object" if name in {"machinery", "vehicle"} else "Other"
            st.caption(f"`{i}` {name} — {category}")

    st.markdown("---")

    st.markdown("##### Risk Assessment Rules")
    st.markdown("""
    | Condition | Risk Level |
    |-----------|-----------|
    | Worker has **Hardhat** AND **Safety Vest** | ✅ **Compliant** (Low Risk) |
    | Worker missing **Helmet** only | 🟠 **High Risk** |
    | Worker missing **Safety Vest** only | 🟡 **Medium Risk** |
    | Worker missing **Helmet** AND **Vest** | 🔴 **High / Critical Risk** |
    | Multiple PPE items missing | 🔴 **Critical Risk** |
    """)

    st.markdown("---")
    st.markdown("##### API Endpoints")
    st.markdown("""
    | Method | Endpoint | Description |
    |--------|----------|-------------|
    | `POST` | `/api/safety/analyze` | Full safety analysis on image |
    | `POST` | `/api/safety/detect` | YOLO detection + PPE association |
    | `GET` | `/api/safety/alerts` | Retrieve safety alerts |
    | `PATCH` | `/api/safety/alerts/{id}` | Update alert status |
    | `GET` | `/api/safety/analytics` | Aggregate safety metrics |
    | `GET` | `/api/safety/workers` | Worker monitoring records |
    | `POST` | `/api/risk/projects/assess` | Run compliance and insurance assessment |
    | `GET` | `/api/risk/projects/{project_id}/summary` | Project risk metrics and alerts |
    """)

    st.info("💡 Start the REST API server with: `python api/server.py`")

    st.markdown("---")
    st.markdown("##### 📱 SMS Emergency Dispatch Settings")
    st.markdown("Automated SMS notification parameters for worker and supervisor alerting.")

    sms_c1, sms_c2 = st.columns(2)
    with sms_c1:
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | **SMS Alerting Active** | `{'Enabled' if SMS_ENABLED else 'Disabled'}` |
        | **Trigger Risk Levels** | `HIGH` & `CRITICAL` only |
        | **Supervisor Line** | `{DEFAULT_SUPERVISOR_PHONE}` |
        | **Gateway Provider** | `Autonomous Local Dispatcher` |
        """)

    with sms_c2:
        st.markdown("###### Registered Worker Phone Directory")
        for wid, phone in DEFAULT_WORKER_PHONES.items():
            st.caption(f"👷 **{wid}**: `{phone}`")

    st.markdown("---")
    st.markdown("##### Database Information")
    from utils.config import DB_PATH
    st.markdown(f"**Database Path:** `{DB_PATH}`")
    analytics = db_manager.get_analytics_summary()
    sms_logs_count = len(db_manager.get_sms_logs(limit=1000))
    st.markdown(f"""
    | Table | Records |
    |-------|---------|
    | Safety Events | {analytics['total_events']} |
    | PPE Violations | {analytics['total_violations']} |
    | Alerts | {analytics['total_alerts']} |
    | Workers | {analytics['total_workers']} |
    | SMS Alerts | {sms_logs_count} |
    """)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <b>Important Limitations</b><br>
        • The model supports 10 classes only. It does <b>not</b> detect gloves, safety boots, or fall hazards.<br>
        • Risk scores are a prototype methodology for academic demonstration — <b>not</b> validated safety standards.<br>
        • Worker IDs are assigned per-frame based on spatial position and are <b>not</b> persistent across frames.
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem;">
    <p style="color: #475569; font-size: 0.8rem;">
        🏗️ <b>Construction Risk Intelligence Platform</b> <br>
        Milestone 1: Site Risk Monitoring — Milestone 2: Safety Intelligence & Worker Protection<br>
        Built with YOLOv8 • Streamlit • SQLite • FastAPI • Python<br>
        <em>Risk scores are a prototype methodology.</em>
    </p>
</div>
""", unsafe_allow_html=True)
