"""
config.py - Application Configuration

Loads OpenAI-compatible provider settings (base URL, API key, model ID)
and Text-to-Speech (TTS) settings from .env.
Defines Nova's core persona system prompt.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Find the directory where this script lives and load .env from there
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Provider-agnostic LLM settings (compatible with Groq, Ollama, OpenRouter, OpenAI)
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").strip()
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "").strip()
MODEL_ID: str = os.getenv("MODEL_ID", "llama-3.3-70b-versatile").strip()

# Text-to-Speech (TTS) voice setting using Microsoft Edge TTS
# Default: en-US-AriaNeural (natural, clear English voice)
# Alternatives: en-US-GuyNeural, en-GB-SoniaNeural, en-IN-NeerjaNeural
TTS_VOICE: str = os.getenv("TTS_VOICE", "en-US-AriaNeural").strip()

# Path to the SQLite database file
DB_PATH: str = str(BASE_DIR / "nova.db")

# System prompt defining Nova's personality, tone, and operational boundaries
SYSTEM_PROMPT: str = (
    "You are Nova, a friendly, concise, and helpful AI assistant designed for a university student. "
    "You explain complex ideas simply, keep responses focused without unnecessary fluff, "
    "and always maintain an encouraging and polite demeanor."
)


def validate_config() -> tuple[bool, str]:
    """
    Checks if the required configuration (like LLM_API_KEY) is present.
    Allows local Ollama endpoints without an explicit key.
    Returns: (is_valid: bool, error_message: str).
    """
    is_local = "localhost" in LLM_BASE_URL or "127.0.0.1" in LLM_BASE_URL

    if not is_local and (not LLM_API_KEY or "your_" in LLM_API_KEY):
        return False, (
            "API key is missing! Please configure `LLM_API_KEY` in your `.env` file. "
            "You can use Groq (free at https://console.groq.com), OpenRouter, or run Ollama locally."
        )
    return True, ""
