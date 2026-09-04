# 🏗️ Construction Risk Intelligence Platform

## Milestone 1 — Site Risk Monitoring & Hazard Detection

An AI-powered construction site monitoring system that analyses construction-site images/videos, detects site objects and potential hazards using a YOLOv8 object-detection model, calculates a site risk score, and displays the results through an interactive Streamlit dashboard.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Object Detection** | YOLOv8 detects 10 construction-site object classes |
| **Hazard Detection** | Identifies PPE violations and proximity hazards |
| **Risk Scoring** | Transparent 0–100 scoring with configurable weights |
| **Risk Levels** | Low / Medium / High / Critical classification |
| **Image Analysis** | Upload JPG/PNG images for instant analysis |
| **Video Analysis** | Frame-by-frame video processing with trend charts |
| **Safety Recommendations** | Context-aware recommendations based on detections |
| **Analysis History** | Persistent JSON logging of all analyses |
| **Professional Dashboard** | Dark-themed Streamlit UI with gauges and charts |

---

## 🏛️ Architecture

```
Construction Site Image / Video
          ↓
    Input Processing
          ↓
   YOLO Object Detection (YOLOv8n)
          ↓
   Detected Objects + Confidence
          ↓
     Site Risk Agent (Agentic Layer)
          ↓
   Hazard Detection Rules + Proximity Analysis
          ↓
   Risk Score Calculation (0–100)
          ↓
   Risk Level (Low / Medium / High / Critical)
          ↓
   Risk Monitoring Dashboard
```

---

## 📁 Project Structure

```
construction risk mangement/
├── app.py                          # Streamlit dashboard
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── models/
│   ├── best.pt                     # Pre-trained YOLOv8n weights
│   └── results.csv                 # Training metrics (100 epochs)
│
├── data/
│   ├── data.yaml                   # Dataset configuration
│   ├── train/images/ + labels/     # 2,605 training images
│   ├── valid/images/ + labels/     # 114 validation images
│   └── test/images/ + labels/      # 82 test images
│
├── agents/
│   └── site_risk_agent.py          # Site Risk Agent (agentic layer)
│
├── detection/
│   └── detector.py                 # YOLO inference wrapper
│
├── risk/
│   └── risk_scoring.py             # Risk scoring engine
│
├── utils/
│   ├── config.py                   # Central configuration
│   ├── visualization.py            # Matplotlib charts
│   └── logger.py                   # Analysis logging
│
├── outputs/
│   └── analysis_log.json           # Persistent analysis history
│
└── sample_images/                  # Demo test images
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10 or later
- pip package manager

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Verify the model

Ensure `models/best.pt` exists. The pre-trained YOLOv8n model should already be in place.

### Step 3 — Launch the dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

---

## 🎯 Usage

1. **Upload an Image** — Use the sidebar to upload a `.jpg`, `.jpeg`, or `.png` construction-site image.
2. **Upload a Video** — Upload a `.mp4`, `.avi`, or `.mov` video file for frame-by-frame analysis.
3. **Adjust Settings** — Use the confidence threshold slider and video processing controls.
4. **View Results** — The dashboard displays:
   - YOLO bounding-box detections (colour-coded)
   - Risk score gauge (0–100)
   - Risk level indicator
   - Hazard breakdown table
   - Detection statistics chart
   - Safety recommendations
   - Analysis history

---

## ⚙️ Detected Classes

| Class | Type |
|-------|------|
| Hardhat | Safety Equipment ✅ |
| Mask | Safety Equipment ✅ |
| Safety Vest | Safety Equipment ✅ |
| Safety Cone | Safety Equipment ✅ |
| Person | Worker 👷 |
| NO-Hardhat | Hazard ⚠️ |
| NO-Mask | Hazard ⚠️ |
| NO-Safety Vest | Hazard ⚠️ |
| machinery | Heavy Object 🚜 |
| vehicle | Heavy Object 🚛 |

---

## 📊 Risk Scoring Methodology

> **⚠️ Prototype Scoring:** Risk weights and thresholds are a demonstration methodology designed for academic purposes. They are NOT scientifically validated safety standards.

### Hazard Weights

| Hazard | Weight |
|--------|--------|
| NO-Hardhat | +25 |
| NO-Safety Vest | +25 |
| NO-Mask | +5 |

| Machinery / vehicle | Detected object only; not automatically a hazard |

### Risk Levels

| Score Range | Level | Indicator |
|-------------|-------|-----------|
| 0–24 | Low | 🟢 |
| 25–49 | Medium | 🟡 |
| 50–74 | High | 🟠 |
| 75–100 | Critical | 🔴 |

The raw score is normalised against a 100-point ceiling. A single supported
PPE violation contributes 25 points (Medium); multiple violation types add 10
points, multiple affected workers add 5 points per additional worker, and
multiple PPE violations associated with one worker add 10 points. No risk is
inferred solely from a worker being present without a detected PPE violation.

Machinery and vehicles are reported as detected objects only. Milestone 1 does
not claim that they are hazards merely because they appear near a worker.

### Video Risk Aggregation

Video analysis scores each sampled frame independently from its detected
objects. Worker and hazard metrics are reported as decimal averages and also
include the maximum count observed in one frame, so short-lived detections are
not displayed as zero. The video risk level uses the larger of the average
score and the average/maximum midpoint (the peak-aware score). This preserves
average exposure while retaining evidence of a severe PPE peak.

The model has no footwear, fall, scaffold, or working-at-height classes. The
system can report workers and PPE violations that the model detects, but it
cannot claim those untrained hazard types from pixels. Safety footwear and
working-at-height review therefore remain manual until the model is fine-tuned
on a dataset containing those labels.

---

## 📜 Dataset

- **Source:** [Roboflow Construction Site Safety](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety)
- **License:** CC BY 4.0
- **Images:** ~2,800 annotated construction-site images
- **Format:** YOLO (bounding box annotations)

---

## 🛠️ Technology Stack

| Technology | Purpose |
|------------|---------|
| Python 3.12 | Core language |
| Ultralytics YOLOv8 | Object detection |
| OpenCV | Image/video processing |
| Streamlit | Dashboard UI |
| Matplotlib | Visualisation charts |
| Pandas | Data handling |
| NumPy | Numerical operations |
| JSON | Lightweight result storage |

---

## 📌 Academic Demonstration

This project demonstrates:

1. **Input** — Construction-site image/video
2. **AI Detection** — YOLO detects construction objects
3. **Agentic Layer** — Site Risk Agent interprets detections and determines risk
4. **Decision** — Risk-scoring engine calculates a 0–100 site risk score
5. **Output** — Dashboard presents hazards, risk score, risk level, and recommendations

Every risk decision shown on the dashboard is traceable to actual YOLO detections and configured scoring rules.

---

*Built as Milestone 1 of the Agentic Construction Risk Intelligence Platform.*
