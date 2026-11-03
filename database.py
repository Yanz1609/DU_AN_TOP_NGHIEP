
import sqlite3
from datetime import datetime

DB_NAME = "drowsiness.db"


def init_database():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                ear REAL,
                mar REAL,
                created_at TEXT NOT NULL
            )
        """)


def save_alert(event_type, ear=None, mar=None):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute(
            """
            INSERT INTO alerts (event_type, ear, mar, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                event_type,
                ear,
                mar,
                datetime.now().isoformat(timespec="seconds")
            )
        )
