"""
Database Manager for Safety Intelligence Platform
===================================================
Provides persistent SQLite storage for safety events, PPE violations,
alerts, and worker monitoring records.
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import DB_PATH


class SafetyDBManager:
    """
    Manages SQLite database connections and CRUD operations for Milestone 2.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_tables()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        """Create database tables if they do not exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Safety events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS safety_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source TEXT,
                    worker_count INTEGER DEFAULT 0,
                    compliant_workers INTEGER DEFAULT 0,
                    violations_count INTEGER DEFAULT 0,
                    risk_level TEXT,
                    violation_type TEXT,
                    raw_json TEXT
                )
            """)

            # PPE violations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ppe_violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER,
                    timestamp TEXT NOT NULL,
                    worker_id TEXT,
                    violation_type TEXT,
                    missing_ppe TEXT,
                    risk_level TEXT,
                    confidence REAL,
                    FOREIGN KEY(event_id) REFERENCES safety_events(id)
                )
            """)

            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    worker_id TEXT,
                    confidence REAL,
                    status TEXT DEFAULT 'New',
                    cooldown_key TEXT
                )
            """)

            # Milestone 3 project-level alert fields. Existing Milestone 2
            # alert records remain compatible through nullable migration columns.
            existing_alert_columns = {
                row[1] for row in cursor.execute("PRAGMA table_info(alerts)").fetchall()
            }
            for column, definition in {
                "project_id": "TEXT",
                "alert_type": "TEXT",
                "severity": "TEXT",
                "created_at": "TEXT",
            }.items():
                if column not in existing_alert_columns:
                    cursor.execute(f"ALTER TABLE alerts ADD COLUMN {column} {definition}")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS compliance_checks (
                    compliance_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    regulation_name TEXT NOT NULL,
                    compliance_status TEXT NOT NULL,
                    checked_at TEXT NOT NULL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS insurance_cases (
                    case_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    claim_type TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    status TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)

            # Upgrade insurance_cases created by earlier Milestone 3 versions.
            # Nullable/defaulted columns keep existing insurance records intact.
            existing_insurance_columns = {
                row[1] for row in cursor.execute("PRAGMA table_info(insurance_cases)").fetchall()
            }
            for column, definition in {
                "project_id": "TEXT",
                "status": "TEXT DEFAULT 'Open'",
                "timestamp": "TEXT",
            }.items():
                if column not in existing_insurance_columns:
                    cursor.execute(f"ALTER TABLE insurance_cases ADD COLUMN {column} {definition}")
            cursor.execute(
                "UPDATE insurance_cases SET timestamp = ? WHERE timestamp IS NULL",
                (datetime.now().isoformat(),),
            )

            # Worker tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    ppe_status TEXT NOT NULL,
                    missing_ppe TEXT,
                    risk_level TEXT NOT NULL,
                    detection_count INTEGER DEFAULT 1,
                    violation_count INTEGER DEFAULT 0
                )
            """)

            # SMS Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sms_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_id TEXT,
                    worker_id TEXT,
                    phone_number TEXT,
                    risk_level TEXT,
                    message TEXT,
                    status TEXT,
                    provider TEXT
                )
            """)

            conn.commit()

    # ------------------------------------------------------------------
    # Safety Events
    # ------------------------------------------------------------------
    def log_safety_event(
        self,
        timestamp: str,
        source: str,
        worker_count: int,
        compliant_workers: int,
        violations_count: int,
        risk_level: str,
        violation_type: str,
        raw_json: Optional[Dict] = None,
    ) -> int:
        json_str = json.dumps(raw_json) if raw_json else "{}"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO safety_events (timestamp, source, worker_count, compliant_workers, violations_count, risk_level, violation_type, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (timestamp, source, worker_count, compliant_workers, violations_count, risk_level, violation_type, json_str)
            )
            conn.commit()
            return cursor.lastrowid

    def get_recent_events(self, limit: int = 50) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM safety_events ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # PPE Violations
    # ------------------------------------------------------------------
    def log_ppe_violation(
        self,
        event_id: Optional[int],
        timestamp: str,
        worker_id: str,
        violation_type: str,
        missing_ppe: List[str],
        risk_level: str,
        confidence: float,
    ) -> int:
        missing_str = json.dumps(missing_ppe)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO ppe_violations (event_id, timestamp, worker_id, violation_type, missing_ppe, risk_level, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (event_id, timestamp, worker_id, violation_type, missing_str, risk_level, confidence)
            )
            conn.commit()
            return cursor.lastrowid

    def get_violations(self, limit: int = 100) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ppe_violations ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            res = []
            for r in rows:
                item = dict(r)
                try:
                    item["missing_ppe"] = json.loads(item["missing_ppe"])
                except Exception:
                    pass
                res.append(item)
            return res

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------
    def save_alert(self, alert_data: Dict) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO alerts (alert_id, timestamp, violation_type, risk_level, worker_id, confidence, status, cooldown_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert_data["alert_id"],
                    alert_data["timestamp"],
                    alert_data["violation_type"],
                    alert_data["risk_level"],
                    alert_data.get("worker_id", "Unknown"),
                    alert_data.get("confidence", 0.0),
                    alert_data.get("status", "New"),
                    alert_data.get("cooldown_key", ""),
                )
            )
            conn.commit()

    def update_alert_status(self, alert_id: str, status: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE alerts SET status = ? WHERE alert_id = ?",
                (status, alert_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_alerts(self, status: Optional[str] = None, limit: int = 100) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute(
                    "SELECT * FROM alerts WHERE status = ? ORDER BY timestamp DESC LIMIT ?",
                    (status, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Milestone 3 Compliance, Insurance, and Project Alerts
    # ------------------------------------------------------------------
    def save_compliance_check(self, check: Dict) -> str:
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO compliance_checks
                    (compliance_id, project_id, regulation_name, compliance_status, checked_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    check["compliance_id"],
                    check["project_id"],
                    check["regulation_name"],
                    check["compliance_status"],
                    check["checked_at"],
                ),
            )
        return check["compliance_id"]

    def get_compliance_checks(self, project_id: Optional[str] = None) -> List[Dict]:
        with self.get_connection() as conn:
            if project_id:
                rows = conn.execute(
                    "SELECT * FROM compliance_checks WHERE project_id = ? ORDER BY checked_at DESC",
                    (project_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM compliance_checks ORDER BY checked_at DESC"
                ).fetchall()
            return [dict(row) for row in rows]

    def save_insurance_case(self, case: Dict) -> str:
        with self.get_connection() as conn:
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(insurance_cases)").fetchall()
            }
            values = {
                "case_id": case["case_id"],
                "project_id": case["project_id"],
                "claim_type": case["claim_type"],
                "risk_score": case["risk_score"],
                "status": case["status"],
                "timestamp": case.get("timestamp") or datetime.now().isoformat(),
            }
            insert_columns = [column for column in values if column in columns]
            placeholders = ", ".join("?" for _ in insert_columns)
            column_sql = ", ".join(insert_columns)
            conn.execute(
                f"INSERT OR REPLACE INTO insurance_cases ({column_sql}) VALUES ({placeholders})",
                tuple(values[column] for column in insert_columns),
            )
        return case["case_id"]

    def get_insurance_cases(self, project_id: Optional[str] = None) -> List[Dict]:
        with self.get_connection() as conn:
            if project_id:
                rows = conn.execute(
                    "SELECT * FROM insurance_cases WHERE project_id = ? ORDER BY risk_score DESC",
                    (project_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM insurance_cases ORDER BY risk_score DESC"
                ).fetchall()
            return [dict(row) for row in rows]

    def get_project_risk_summary(self, project_id: Optional[str] = None) -> Dict:
        checks = self.get_compliance_checks(project_id)
        cases = self.get_insurance_cases(project_id)
        alerts = [
            alert for alert in self.get_alerts()
            if not project_id or alert.get("project_id") == project_id
        ]
        compliant = sum(check["compliance_status"] == "Compliant" for check in checks)
        readiness = round((compliant / max(1, len(checks))) * 100.0, 1)
        highest_case_score = max((case["risk_score"] for case in cases), default=0.0)
        return {
            "project_id": project_id,
            "audit_readiness_score": readiness,
            "open_violations": sum(check["compliance_status"] != "Compliant" for check in checks),
            "insurance_risk_score": highest_case_score,
            "insurance_exposure": (
                "High" if highest_case_score >= 70 else
                "Medium" if highest_case_score >= 35 else "Low"
            ),
            "alerts": alerts,
        }

    def get_project_compliance_report(self, project_id: str) -> Dict:
        """Return the persisted compliance and insurance audit report."""
        checks = self.get_compliance_checks(project_id)
        cases = self.get_insurance_cases(project_id)
        summary = self.get_project_risk_summary(project_id)
        return {
            "project_id": project_id,
            "summary": summary,
            "compliance_checks": checks,
            "insurance_cases": cases,
            "report_generated_at": datetime.now().isoformat(),
        }

    def save_project_alert(self, alert: Dict) -> str:
        now = alert.get("created_at", datetime.now().isoformat())
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO alerts
                    (alert_id, project_id, alert_type, severity, created_at,
                     timestamp, violation_type, risk_level, status, cooldown_key, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert["alert_id"],
                    alert["project_id"],
                    alert["alert_type"],
                    alert["severity"],
                    now,
                    now,
                    alert["alert_type"],
                    alert["severity"],
                    alert.get("status", "New"),
                    alert.get("cooldown_key", ""),
                    float(alert.get("confidence", 1.0)),
                ),
            )
        return alert["alert_id"]

    def check_alert_cooldown(self, cooldown_key: str, cooldown_seconds: float) -> bool:
        """Returns True if alert is in cooldown (should be suppressed)."""
        if not cooldown_key:
            return False
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT timestamp FROM alerts WHERE cooldown_key = ? ORDER BY timestamp DESC LIMIT 1",
                (cooldown_key,)
            )
            row = cursor.fetchone()
            if row:
                try:
                    last_time = datetime.fromisoformat(row["timestamp"])
                    elapsed = (datetime.now() - last_time).total_seconds()
                    if elapsed < cooldown_seconds:
                        return True
                except Exception:
                    pass
        return False

    # ------------------------------------------------------------------
    # Worker Tracking Records
    # ------------------------------------------------------------------
    def update_worker_record(
        self,
        worker_id: str,
        timestamp: str,
        ppe_status: str,
        missing_ppe: List[str],
        risk_level: str,
        is_violation: bool = False,
    ) -> None:
        missing_str = json.dumps(missing_ppe)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM workers WHERE worker_id = ?", (worker_id,))
            row = cursor.fetchone()
            if row:
                det_count = row["detection_count"] + 1
                viol_count = row["violation_count"] + (1 if is_violation else 0)
                cursor.execute(
                    """
                    UPDATE workers
                    SET last_seen = ?, ppe_status = ?, missing_ppe = ?, risk_level = ?, detection_count = ?, violation_count = ?
                    WHERE worker_id = ?
                    """,
                    (timestamp, ppe_status, missing_str, risk_level, det_count, viol_count, worker_id)
                )
            else:
                viol_count = 1 if is_violation else 0
                cursor.execute(
                    """
                    INSERT INTO workers (worker_id, first_seen, last_seen, ppe_status, missing_ppe, risk_level, detection_count, violation_count)
                    VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                    """,
                    (worker_id, timestamp, timestamp, ppe_status, missing_str, risk_level, viol_count)
                )
            conn.commit()

    def get_workers(self, limit: int = 100) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM workers ORDER BY last_seen DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            res = []
            for r in rows:
                item = dict(r)
                try:
                    item["missing_ppe"] = json.loads(item["missing_ppe"])
                except Exception:
                    pass
                res.append(item)
            return res

    # ------------------------------------------------------------------
    # Analytics & Aggregates
    # ------------------------------------------------------------------
    def get_analytics_summary(self) -> Dict:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) as cnt FROM safety_events")
            total_events = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM workers")
            total_workers = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM workers WHERE ppe_status = 'Compliant'")
            compliant_workers = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM workers WHERE ppe_status != 'Compliant'")
            non_compliant_workers = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM ppe_violations")
            total_violations = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM alerts WHERE status = 'New'")
            active_alerts = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM alerts")
            total_alerts = cursor.fetchone()["cnt"]

            if total_workers > 0:
                compliance_pct = (compliant_workers / total_workers) * 100.0
                safety_score = round(compliance_pct, 1)
            else:
                safety_score = 100.0

            cursor.execute("SELECT missing_ppe FROM ppe_violations")
            violation_types = {}
            for row in cursor.fetchall():
                try:
                    missing = set(json.loads(row["missing_ppe"]))
                except (TypeError, ValueError, json.JSONDecodeError):
                    missing = set()

                if "helmet" in missing and "vest" in missing:
                    label = "Missing helmet + vest"
                elif "helmet" in missing:
                    label = "Missing helmet"
                elif "vest" in missing:
                    label = "Missing vest"
                elif missing:
                    label = "Missing " + ", ".join(sorted(missing))
                else:
                    label = "Other PPE violation"
                violation_types[label] = violation_types.get(label, 0) + 1

            cursor.execute("SELECT risk_level, COUNT(*) as cnt FROM ppe_violations GROUP BY risk_level")
            risk_distribution = {r["risk_level"]: r["cnt"] for r in cursor.fetchall()}

            return {
                "total_events": total_events,
                "total_workers": total_workers,
                "compliant_workers": compliant_workers,
                "non_compliant_workers": non_compliant_workers,
                "compliance_percentage": round((compliant_workers / max(1, total_workers)) * 100, 1),
                "total_violations": total_violations,
                "active_alerts": active_alerts,
                "total_alerts": total_alerts,
                "safety_score": safety_score,
                "violation_types": violation_types,
                "risk_distribution": risk_distribution,
            }

    # ------------------------------------------------------------------
    # SMS Alert Logging
    # ------------------------------------------------------------------
    def log_sms_alert(
        self,
        alert_id: str,
        worker_id: str,
        phone_number: str,
        risk_level: str,
        message: str,
        status: str = "Delivered",
        provider: str = "Local Dispatcher",
    ) -> int:
        now_iso = datetime.now().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO sms_alerts (timestamp, alert_id, worker_id, phone_number, risk_level, message, status, provider)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (now_iso, alert_id, worker_id, phone_number, risk_level, message, status, provider)
            )
            conn.commit()
            return cursor.lastrowid

    def get_sms_logs(self, limit: int = 50) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sms_alerts ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]


