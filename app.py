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
    TWILIO_ACCOUNT_SID, TWILIO_FROM_NUMBER,
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
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
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
        font-weight: 800;
        margin: 0.3rem 0;
    }
    .metric-card .label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Section headers */
    .section-header {
        color: #e0e7ff;
        font-size: 1.3rem;
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
        min-height: 128px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        text-align: left;
    }
    .overview-card .description {
        color: #718096;
        font-size: 0.78rem;
        margin-top: 0.4rem;
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
        ["🏠 Dashboard", "🛡️ Safety Monitoring", "🔍 Live Detection",
         "🚨 Alerts", "📊 Analytics", "👷 Workers", "⚙️ Configuration"],
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
    """Play one short beep for each newly displayed high-severity alert batch."""
    urgent_alerts = [
        alert for alert in alerts
        if str(alert.get("risk_level", "")).upper() in {"HIGH", "CRITICAL"}
    ]
    if not urgent_alerts:
        return

    seen_alert_ids = st.session_state.setdefault("beeped_alert_ids", set())
    new_alerts = [
        alert for alert in urgent_alerts
        if alert.get("alert_id") not in seen_alert_ids
    ]
    if not new_alerts:
        return

    seen_alert_ids.update(alert.get("alert_id") for alert in new_alerts)
    sample_rate = 44100
    duration_seconds = 0.18
    frequency = 880
    amplitude = 12000
    frames = bytearray()
    for sample_index in range(int(sample_rate * duration_seconds)):
        value = int(amplitude * np.sin(2 * np.pi * frequency * sample_index / sample_rate))
        frames.extend(value.to_bytes(2, byteorder="little", signed=True))

    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate)
        audio_file.writeframes(frames)

    audio_data = base64.b64encode(audio_buffer.getvalue()).decode("ascii")
    st.markdown(
        f'<audio autoplay><source src="data:audio/wav;base64,{audio_data}" type="audio/wav"></audio>',
        unsafe_allow_html=True,
    )


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
        bar_width = max(4, int((width / 100) * 100))
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
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(alert.get('timestamp', ''))[:19])}</td>"
            f"<td>{html.escape(str(alert.get('worker_id', 'Unknown')))}</td>"
            f"<td>{html.escape(str(alert.get('violation_type', 'Safety violation')))}</td>"
            f'<td><span class="risk-pill {risk_class}">{html.escape(risk)}</span></td>'
            f"<td>{float(alert.get('confidence', 0)):.0%}</td>"
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

    st.markdown("")

    st.markdown("")

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
                    st.markdown(f"""
                    <div class="alert-card {sev}">
                        <div class="alert-title">{alert.get('risk_level', 'N/A')} RISK ALERT</div>
                        <div class="alert-meta">
                            {alert.get('violation_type', '')} &nbsp;│&nbsp;
                            🕐 {alert.get('timestamp', '')[:19]} &nbsp;│&nbsp;
                            📊 Confidence: {alert.get('confidence', 0):.0%} &nbsp;│&nbsp;
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

            st.markdown(f"""
            <div class="alert-card {sev}">
                <div class="alert-title">{alert.get('risk_level', 'N/A')} RISK ALERT — {alert_id}</div>
                <div class="alert-meta">
                    ⚠️ {alert.get('violation_type', 'Safety Violation')} <br>
                    🕐 Time: {alert.get('timestamp', '')[:19]} &nbsp;│&nbsp;
                    👷 Worker: {alert.get('worker_id', 'N/A')} &nbsp;│&nbsp;
                    📊 Confidence: {alert.get('confidence', 0):.0%} &nbsp;│&nbsp;
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

    st.markdown("")

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
        | **Gateway Provider** | `{'Twilio REST' if TWILIO_ACCOUNT_SID else 'Local Dispatcher (Simulator)'}` |
        """)

    with sms_c2:
        st.markdown("###### Registered Worker Phone Directory")
        for wid, phone in DEFAULT_WORKER_PHONES.items():
            st.caption(f"👷 **{wid}**: `{phone}`")

    with st.expander("🔑 Configure Twilio SMS Credentials", expanded=not bool(os.getenv("TWILIO_ACCOUNT_SID"))):
        st.markdown("""
        To receive real cellular SMS on your mobile phone (**`+91 6370671276`**):
        1. Open your [Twilio Console](https://console.twilio.com).
        2. Copy your **Account SID**, **Auth Token**, and **Twilio Phone Number**.
        3. Save below to transmit live messages to your phone carrier.
        """)
        tw_sid_input = st.text_input("Twilio Account SID", value=os.getenv("TWILIO_ACCOUNT_SID", ""), type="password")
        tw_token_input = st.text_input("Twilio Auth Token", value=os.getenv("TWILIO_AUTH_TOKEN", ""), type="password")
        tw_from_input = st.text_input("Twilio Phone Number (Sender)", value=os.getenv("TWILIO_FROM_NUMBER", ""), placeholder="+1234567890")

        if st.button("💾 Save Twilio Credentials"):
            env_path = os.path.join(PROJECT_ROOT, ".env")
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(f"TWILIO_ACCOUNT_SID={tw_sid_input.strip()}\n")
                f.write(f"TWILIO_AUTH_TOKEN={tw_token_input.strip()}\n")
                f.write(f"TWILIO_FROM_NUMBER={tw_from_input.strip()}\n")
            os.environ["TWILIO_ACCOUNT_SID"] = tw_sid_input.strip()
            os.environ["TWILIO_AUTH_TOKEN"] = tw_token_input.strip()
            os.environ["TWILIO_FROM_NUMBER"] = tw_from_input.strip()
            st.success("✅ Twilio credentials saved! Real SMS dispatch is now activated.")
            st.rerun()

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
