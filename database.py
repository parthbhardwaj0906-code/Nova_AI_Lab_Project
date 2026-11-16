"""
database.py - SQLite Database Management

Provides persistent storage for:
1. Chat History: Keeps conversation turns across app restarts.
2. User Notes: Personal memos saved by Nova.
3. Reminders: Tasks with optional due times.
4. User Preferences: Long-term memory profile (e.g. name, tone, major).

Automatically creates nova.db on first run if it does not already exist.
"""

import sqlite3
from typing import List, Dict, Any
from config import DB_PATH


def get_connection() -> sqlite3.Connection:
    """
    Creates and returns a connection to the SQLite database.
    Enables row factory so query results can be accessed like dictionaries.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Initializes the database schema by creating required tables if they do not exist.
    Called automatically on application startup.
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
    """Saves a single message (either from 'user' or 'assistant') to the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
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
            (title, content),
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
def save_reminder(task: str, due_time: str) -> int:
    """Saves a reminder task and returns the generated reminder ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO reminders (task, due_time) VALUES (?, ?)",
            (task, due_time),
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
    Uses UPSERT (INSERT ... ON CONFLICT DO UPDATE) so existing keys are modified.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO user_preferences (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
            """,
            (key.strip().lower(), value.strip()),
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
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_preferences WHERE key = ?", (key.strip().lower(),))
        conn.commit()


# Automatically initialize tables when this module is imported
init_db()
