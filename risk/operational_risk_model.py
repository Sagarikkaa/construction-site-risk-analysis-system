from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "archive_2"
    / "construction_project_dataset.csv"
)


class OperationalRiskModel:
    """Utility for reading the auxiliary construction operations dataset.

    This dataset is tabular and complements the image-based YOLO safety model.
    It does not replace the YOLO dataset, but it adds project-level risk context
    such as cost deviation, worker counts, material usage, and operational risk.
    """

    def __init__(self, csv_path: Optional[str | Path] = None):
        self.csv_path = Path(csv_path) if csv_path else DEFAULT_DATASET_PATH
        self.df = self._load_dataset()

    def _load_dataset(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            return pd.DataFrame()

        df = pd.read_csv(self.csv_path)
        if df.empty:
            return df

        # Standardize timestamp column if present so the dataset can be explored cleanly.
        if "timestamp" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            except Exception:
                pass

        return df

    def is_available(self) -> bool:
        return not self.df.empty

    def summary(self) -> Dict:
        if self.df.empty:
            return {
                "available": False,
                "dataset_name": self.csv_path.name,
                "row_count": 0,
                "avg_risk_score": 0,
                "max_risk_score": 0,
                "risk_level": "Unavailable",
                "warning": "No operational CSV dataset was found.",
            }

        risk_col = "risk_score" if "risk_score" in self.df.columns else None
        if risk_col is None:
            return {
                "available": False,
                "dataset_name": self.csv_path.name,
                "row_count": len(self.df),
                "avg_risk_score": 0,
                "max_risk_score": 0,
                "risk_level": "Unavailable",
                "warning": "The CSV does not contain a risk_score column.",
            }

        numeric_risk = pd.to_numeric(self.df[risk_col], errors="coerce").dropna()
        avg_risk = float(numeric_risk.mean()) if not numeric_risk.empty else 0.0
        max_risk = float(numeric_risk.max()) if not numeric_risk.empty else 0.0

        return {
            "available": True,
            "dataset_name": self.csv_path.name,
            "row_count": int(len(self.df)),
            "avg_risk_score": round(avg_risk, 1),
            "max_risk_score": round(max_risk, 1),
            "risk_level": self._risk_level(avg_risk),
            "warning": "Operational CSV is used as a project-level risk complement to the image detector.",
        }

    def recent_rows(self, limit: int = 10) -> List[Dict]:
        if self.df.empty:
            return []
        return self.df.head(limit).to_dict(orient="records")

    def _risk_level(self, score: float) -> str:
        if score >= 75:
            return "Critical"
        if score >= 50:
            return "High"
        if score >= 25:
            return "Medium"
        return "Low"
