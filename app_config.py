"""
app_config.py - Nova Application Configuration

Loads settings from .env (locally) or Streamlit Secrets (on Streamlit Cloud).
Using 'app_config.py' prevents name collisions with external 'config' packages in Linux/Cloud environments.
"""

import os
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Safely load local .env if python-dotenv is present and file exists
try:
    from dotenv import load_dotenv
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except Exception:
    pass


def get_setting(key: str, default: str = "") -> str:
    """
    Retrieves a configuration setting with fallback:
    1. OS environment variables (or .env).
    2. Streamlit Cloud Secrets (st.secrets).
    3. The provided default value.
    """
    val = os.getenv(key)
    if val is not None and val.strip():
        return val.strip()

    # Fallback to Streamlit secrets when running on Streamlit Cloud
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            secret_val = st.secrets[key]
            if secret_val:
                return str(secret_val).strip()
    except Exception:
        pass

    return default


# Provider-agnostic LLM configuration (Groq, Ollama, OpenRouter, OpenAI, etc.)
LLM_BASE_URL: str = get_setting("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY: str = get_setting("LLM_API_KEY", "")
MODEL_ID: str = get_setting("MODEL_ID", "llama-3.3-70b-versatile")

# Edge-TTS voice name for optional voice narration
# Default: en-US-AriaNeural (natural, clear English voice)
TTS_VOICE: str = get_setting("TTS_VOICE", "en-US-AriaNeural")

# SQLite database file path
DB_PATH: str = str(BASE_DIR / "nova.db")

# Core persona system prompt defining Nova's tone and behavior
SYSTEM_PROMPT: str = (
    "You are Nova, a friendly, concise, and helpful AI assistant designed for a university student. "
    "You explain complex ideas simply, keep responses focused without unnecessary fluff, "
    "and always maintain an encouraging and polite demeanor."
)


def validate_config() -> tuple[bool, str]:
    """
    Validates essential configuration settings.
    Allows local Ollama endpoints without an explicit API key.
    """
    is_local = "localhost" in LLM_BASE_URL or "127.0.0.1" in LLM_BASE_URL

    if not is_local and (not LLM_API_KEY or "your_" in LLM_API_KEY):
        return False, (
            "API key is missing! Please set `LLM_API_KEY` in your `.env` file or Streamlit Cloud Secrets. "
            "You can get a free key from Groq (https://console.groq.com) or run Ollama locally."
        )
    return True, ""
