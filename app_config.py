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
# Default is OpenRouter pointing to gpt-oss-120b — change to any OpenAI-compatible provider
LLM_BASE_URL: str = get_setting("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY: str = get_setting("LLM_API_KEY", "")
MODEL_ID: str = get_setting("MODEL_ID", "openai/gpt-oss-120b")

# Models that support image/vision inputs via the OpenAI multimodal message format.
# If the active model is NOT in this list, the UI will warn the user before sending images.
VISION_MODELS: set = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4-vision-preview",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-opus",
    "anthropic/claude-3.5-haiku",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4-turbo",
    "google/gemini-pro-vision",
    "google/gemini-flash-1.5",
    "google/gemini-pro-1.5",
    "meta-llama/llama-3.2-11b-vision-instruct",
    "meta-llama/llama-3.2-90b-vision-instruct",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview",
    "llava",
    "bakllava",
    "moondream",
}


def is_vision_model(model_id: str) -> bool:
    """
    Returns True if the configured model supports image (vision) inputs.
    Checks against a known list of vision-capable model IDs.
    If your model is not listed but supports vision, add it to VISION_MODELS above.
    """
    model_lower = model_id.lower()
    return any(vm.lower() in model_lower for vm in VISION_MODELS)

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
