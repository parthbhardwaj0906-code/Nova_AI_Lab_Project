"""
database.py - SQLite Database Management for Nova

Provides persistent storage for:
1. Chat History: Preserves conversation messages across app restarts.
2. User Notes: Personal notes and memos saved by Nova.
3. Reminders: Task reminders with optional due dates/times.
4. User Preferences: Long-term profile memory (e.g., name, response style, major).

Handles database initialization and backward compatibility automatically.
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
    Initializes the database schema by creating required tables if they do not exist.
    """
    with get_connection() as conn:
        cursor = conn.cursor()

        # 1. Chat message history
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # 2. Personal notes
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # 3. Tasks and reminders
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT NOT NULL,
                due_time TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # 4. Long-term user preferences & traits
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.commit()


# ==============================================================================
# Chat History Operations
# ==============================================================================
def save_message(role: str, content: str) -> None:
    """Saves a single message (user or assistant) to the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        # Check if messages table has a legacy user_id column
        cols = [r[1] for r in cursor.execute("PRAGMA table_info(messages)").fetchall()]
        if "user_id" in cols:
            cursor.execute(
                "INSERT INTO messages (user_id, role, content) VALUES ('default_user', ?, ?)",
                (role, content),
            )
        else:
            cursor.execute(
                "INSERT INTO messages (role, content) VALUES (?, ?)",
                (role, content),
            )
        conn.commit()


def load_messages() -> List[Dict[str, str]]:
    """Retrieves all chat messages in chronological order."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT role, content FROM messages ORDER BY id ASC")
        rows = cursor.fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]


def clear_chat_history() -> None:
    """Deletes all messages from the messages table to reset the conversation."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages")
        conn.commit()


# ==============================================================================
# Notes Operations
# ==============================================================================
def save_note(title: str, content: str) -> int:
    """Saves a note to the database and returns the generated note ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO notes (title, content) VALUES (?, ?)",
            (title.strip(), content.strip()),
        )
        conn.commit()
        return cursor.lastrowid


def get_notes() -> List[Dict[str, Any]]:
    """Retrieves all saved notes in reverse chronological order (newest first)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, content, created_at FROM notes ORDER BY id DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def delete_note(note_id: int) -> None:
    """Deletes a note by its ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        conn.commit()


# ==============================================================================
# Reminders Operations
# ==============================================================================
def save_reminder(task: str, due_time: str = "") -> int:
    """Saves a reminder task and returns the generated reminder ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO reminders (task, due_time) VALUES (?, ?)",
            (task.strip(), due_time.strip()),
        )
        conn.commit()
        return cursor.lastrowid


def get_reminders() -> List[Dict[str, Any]]:
    """Retrieves all reminders in reverse chronological order."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, task, due_time, created_at FROM reminders ORDER BY id DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def delete_reminder(reminder_id: int) -> None:
    """Deletes a reminder by its ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
        conn.commit()


# ==============================================================================
# User Preferences (Long-term Memory) Operations
# ==============================================================================
def set_preference(key: str, value: str) -> None:
    """
    Saves or updates a user preference key-value pair.
    Handles tables with or without user_id transparently.
    """
    clean_key = key.strip().lower()
    clean_val = value.strip()
    with get_connection() as conn:
        cursor = conn.cursor()
        cols = [r[1] for r in cursor.execute("PRAGMA table_info(user_preferences)").fetchall()]
        if "user_id" in cols:
            cursor.execute(
                """
                INSERT INTO user_preferences (user_id, key, value) VALUES ('default_user', ?, ?)
                ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
                """,
                (clean_key, clean_val),
            )
        else:
            cursor.execute(
                """
                INSERT INTO user_preferences (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
                """,
                (clean_key, clean_val),
            )
        conn.commit()


def get_preferences() -> Dict[str, str]:
    """Retrieves all user preferences as a key-value dictionary."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM user_preferences ORDER BY key ASC")
        rows = cursor.fetchall()
        return {row["key"]: row["value"] for row in rows}


def delete_preference(key: str) -> None:
    """Deletes a specific user preference by key."""
    clean_key = key.strip().lower()
    with get_connection() as conn:
        cursor = conn.cursor()
        cols = [r[1] for r in cursor.execute("PRAGMA table_info(user_preferences)").fetchall()]
        if "user_id" in cols:
            cursor.execute("DELETE FROM user_preferences WHERE user_id = 'default_user' AND key = ?", (clean_key,))
        else:
            cursor.execute("DELETE FROM user_preferences WHERE key = ?", (clean_key,))
        conn.commit()


# Automatically initialize schema on module import
init_db()
