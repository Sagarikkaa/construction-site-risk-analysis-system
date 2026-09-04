"""
Analysis Logger
================
Persists per-analysis records to a JSON log file so the dashboard can
display recent analysis history.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import LOG_FILE, OUTPUT_DIR


class AnalysisLogger:
    """Append-only JSON-lines logger for analysis records."""

    def __init__(self, log_path: str = LOG_FILE):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        # Create the file if it doesn't exist
        if not os.path.isfile(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def log_analysis(
        self,
        filename: str,
        total_objects: int,
        hazard_count: int,
        risk_score: int,
        risk_level: str,
        worker_count: int = 0,
        analysis_type: str = "image",
    ) -> Dict:
        """
        Append a new analysis record and return it.

        Parameters
        ----------
        filename : str        – Name of the analysed file.
        total_objects : int    – Total detected objects.
        hazard_count : int     – Number of hazards detected.
        risk_score : int       – Normalised 0–100 score.
        risk_level : str       – Low / Medium / High / Critical.
        worker_count : int     – Number of workers detected.
        analysis_type : str    – "image" or "video".
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "analysis_type": analysis_type,
            "total_objects": total_objects,
            "hazard_count": hazard_count,
            "worker_count": worker_count,
            "risk_score": risk_score,
            "risk_level": risk_level,
        }

        records = self._read_all()
        records.append(record)

        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        return record

    def get_history(self, limit: Optional[int] = 20) -> List[Dict]:
        """Return the most recent *limit* analysis records (newest first)."""
        records = self._read_all()
        records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        if limit:
            return records[:limit]
        return records

    def _read_all(self) -> List[Dict]:
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, FileNotFoundError):
            return []
