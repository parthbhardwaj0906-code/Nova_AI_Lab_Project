"""
database.py - SQLite Database Management for Nova Health

Provides persistent, user-isolated storage for:
1. Chat History: Conversation turns per user profile.
2. Health Metrics: Numeric biometric logs (weight, sleep, water, blood pressure, etc.).
3. Symptoms: Symptom logs with severity ratings (1-10) and notes for doctor sharing.
4. Medication Reminders: Medication schedules and user notes.
5. User Preferences: Personal health profile, goals, age, and sex.

Automatically creates and migrates nova.db on first run.
"""

import sqlite3
from typing import List, Dict, Any, Optional
from app_config import DB_PATH


def get_connection() -> sqlite3.Connection:
    """
    Creates and returns a connection to the SQLite database.
    Row factory is enabled so columns can be accessed as dictionary keys.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Initializes the database schema by creating required health and chat tables.
    Runs safe schema migrations for existing tables and columns.
    """
    with get_connection() as conn:
        cursor = conn.cursor()

        # 1. Chat messages table (isolated by user_id)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT 'default_user',
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # 2. Health biometric metrics table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS health_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT 'default_user',
                metric TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # 3. Clinical symptom logs table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS symptoms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT 'default_user',
                symptom TEXT NOT NULL,
                severity INTEGER NOT NULL CHECK(severity BETWEEN 1 AND 10),
                notes TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # 4. Medication reminders table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS medication_reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT 'default_user',
                name TEXT NOT NULL,
                time TEXT NOT NULL,
                notes TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # 5. User preferences and profile attributes
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT NOT NULL DEFAULT 'default_user',
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, key)
            )
            """
        )

        # Migration: Ensure user_id column exists in existing tables
        for table in ["messages", "health_metrics", "symptoms", "medication_reminders"]:
            try:
                cols = [r[1] for r in cursor.execute(f"PRAGMA table_info({table})").fetchall()]
                if cols and "user_id" not in cols:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default_user'")
            except Exception:
                pass

        # Migration for user_preferences: Ensure user_id column and composite PK exist
        try:
            pref_cols = [r[1] for r in cursor.execute("PRAGMA table_info(user_preferences)").fetchall()]
            if pref_cols and "user_id" not in pref_cols:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_preferences_new (
                        user_id TEXT NOT NULL DEFAULT 'default_user',
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (user_id, key)
                    )
                    """
                )
                if "updated_at" in pref_cols:
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO user_preferences_new (user_id, key, value, updated_at)
                        SELECT 'default_user', key, value, updated_at FROM user_preferences
                        """
                    )
                else:
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO user_preferences_new (user_id, key, value)
                        SELECT 'default_user', key, value FROM user_preferences
                        """
                    )
                cursor.execute("DROP TABLE user_preferences")
                cursor.execute("ALTER TABLE user_preferences_new RENAME TO user_preferences")
        except Exception:
            try:
                cursor.execute("ALTER TABLE user_preferences ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default_user'")
            except Exception:
                pass

        conn.commit()


# ==============================================================================
# Chat History Operations (Per User)
# ==============================================================================
def save_message(user_id: str, role: str, content: str) -> None:
    """Saves a conversation turn for a specific user profile."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content),
        )
        conn.commit()


def load_messages(user_id: str) -> List[Dict[str, str]]:
    """Retrieves conversation history in chronological order for a specific user profile."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM messages WHERE user_id = ? ORDER BY id ASC",
            (user_id,),
        )
        rows = cursor.fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]


def clear_chat_history(user_id: str) -> None:
    """Clears conversation history for the specified user profile."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        conn.commit()


# ==============================================================================
# Health Metrics Operations
# ==============================================================================
def save_health_metric(user_id: str, metric: str, value: float, unit: str) -> int:
    """Saves a numeric physiological metric (e.g., weight, sleep, water)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO health_metrics (user_id, metric, value, unit) VALUES (?, ?, ?, ?)",
            (user_id, metric.strip().lower(), float(value), unit.strip()),
        )
        conn.commit()
        return cursor.lastrowid


def get_health_metrics(user_id: str, metric: Optional[str] = None, limit: int = 30) -> List[Dict[str, Any]]:
    """Retrieves logged health metrics for a user, optionally filtered by metric type."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if metric:
            cursor.execute(
                """
                SELECT id, metric, value, unit, created_at
                FROM health_metrics
                WHERE user_id = ? AND metric = ?
                ORDER BY id DESC LIMIT ?
                """,
                (user_id, metric.strip().lower(), limit),
            )
        else:
            cursor.execute(
                """
                SELECT id, metric, value, unit, created_at
                FROM health_metrics
                WHERE user_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (user_id, limit),
            )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def delete_health_metric(metric_id: int) -> None:
    """Deletes a health metric record by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM health_metrics WHERE id = ?", (metric_id,))
        conn.commit()


# ==============================================================================
# Symptoms Operations
# ==============================================================================
def save_symptom(user_id: str, symptom: str, severity: int, notes: str = "") -> int:
    """Saves a reported symptom with a 1-10 severity score and descriptive notes."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO symptoms (user_id, symptom, severity, notes) VALUES (?, ?, ?, ?)",
            (user_id, symptom.strip(), int(severity), notes.strip()),
        )
        conn.commit()
        return cursor.lastrowid


def get_symptoms(user_id: str, limit: int = 30) -> List[Dict[str, Any]]:
    """Retrieves logged symptoms for a user in reverse chronological order."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, symptom, severity, notes, created_at
            FROM symptoms
            WHERE user_id = ?
            ORDER BY id DESC LIMIT ?
            """,
            (user_id, limit),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def delete_symptom(symptom_id: int) -> None:
    """Deletes a symptom log entry by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM symptoms WHERE id = ?", (symptom_id,))
        conn.commit()


# ==============================================================================
# Medication Reminders Operations
# ==============================================================================
def save_medication_reminder(user_id: str, name: str, time: str, notes: str = "") -> int:
    """Stores a user-provided medication schedule time (never suggests dosages)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO medication_reminders (user_id, name, time, notes) VALUES (?, ?, ?, ?)",
            (user_id, name.strip(), time.strip(), notes.strip()),
        )
        conn.commit()
        return cursor.lastrowid


def get_medication_reminders(user_id: str) -> List[Dict[str, Any]]:
    """Retrieves all active medication reminders for the user profile."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, name, time, notes, created_at
            FROM medication_reminders
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (user_id,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def delete_medication_reminder(reminder_id: int) -> None:
    """Deletes a medication reminder by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM medication_reminders WHERE id = ?", (reminder_id,))
        conn.commit()


# ==============================================================================
# User Health Preferences (Long-term Profile Memory)
# ==============================================================================
def set_preference(user_id: str, key: str, value: str) -> None:
    """Saves or updates a user health profile attribute (e.g., age, sex, fitness goals)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO user_preferences (user_id, key, value) VALUES (?, ?, ?)
            ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, key.strip().lower(), value.strip()),
        )
        conn.commit()


def get_preferences(user_id: str) -> Dict[str, str]:
    """Retrieves all profile preferences for a given user as a dictionary."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT key, value FROM user_preferences WHERE user_id = ? ORDER BY key ASC",
            (user_id,),
        )
        rows = cursor.fetchall()
        return {row["key"]: row["value"] for row in rows}


def delete_preference(user_id: str, key: str) -> None:
    """Deletes a specific user preference by key."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM user_preferences WHERE user_id = ? AND key = ?",
            (user_id, key.strip().lower()),
        )
        conn.commit()


# Automatically initialize and migrate schema on module load
init_db()
