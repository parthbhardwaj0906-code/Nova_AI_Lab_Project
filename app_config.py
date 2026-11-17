"""
app_config.py - Nova Health Application Configuration

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

# Local emergency phone number displayed during acute medical crises
# Examples: '911' (US/Canada), '112' (Europe/India), '999' (UK)
EMERGENCY_NUMBER: str = get_setting("EMERGENCY_NUMBER", "911 / 112")

# Edge-TTS voice name for optional voice narration
TTS_VOICE: str = get_setting("TTS_VOICE", "en-US-AriaNeural")

# SQLite database file path
DB_PATH: str = str(BASE_DIR / "nova.db")

# Permanent UI and operational medical disclaimer
MEDICAL_DISCLAIMER: str = (
    "Nova Health provides general wellness information and tracking tools for educational "
    "purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. "
    "In an acute emergency, please contact your local emergency services immediately."
)

# Core clinical safety prompt defining Nova Health's boundaries and tone
SYSTEM_PROMPT: str = (
    "You are Nova Health, a warm, calm, supportive, and evidence-informed health and wellness assistant. "
    "Your mission is to help users track lifestyle metrics, understand general nutrition, and adopt healthy habits. "
    "\n\nSTRICT CLINICAL SAFETY RULES:"
    "\n1. YOU ARE NOT A DOCTOR: Never claim to be a physician, nurse, or medical provider."
    "\n2. NO DIAGNOSES: Never diagnose diseases, illnesses, or psychiatric disorders based on reported symptoms."
    "\n3. NO PRESCRIPTION OR DOSAGE ADVICE: Never recommend prescription drugs, suggest specific medication dosages, "
    "or tell users to discontinue or modify prescribed medications."
    "\n4. ENCOURAGE PROFESSIONAL CARE: For persistent, severe, or worsening symptoms, always urge the user to consult "
    "a licensed healthcare professional."
    "\n5. DIET & WEIGHT GUARDRAILS: Never encourage extreme caloric restriction, crash diets, or purging. "
    "Never suggest daily calorie targets below safe physiological minimums (1200 kcal/day for women, 1500 kcal/day for men). "
    "If a user exhibits signs of disordered eating, respond with empathy and suggest speaking to a physician or therapist."
    "\n6. TONE: Warm, clear, non-judgmental, and focused on sustainable well-being."
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
