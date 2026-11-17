"""
config.py - Nova Health Application Configuration

Loads OpenAI-compatible provider settings, emergency contact details,
and Text-to-Speech (TTS) configuration from .env.
Defines Nova Health's clinical boundaries, safety rules, and core persona.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base project directory
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Provider-agnostic LLM configuration (Groq, Ollama, OpenRouter, etc.)
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").strip()
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "").strip()
MODEL_ID: str = os.getenv("MODEL_ID", "llama-3.3-70b-versatile").strip()

# Local emergency phone number displayed during acute medical crises
# Examples: '911' (US/Canada), '112' (Europe/India), '999' (UK)
EMERGENCY_NUMBER: str = os.getenv("EMERGENCY_NUMBER", "911 / 112").strip()

# Edge-TTS voice name for optional voice narration
TTS_VOICE: str = os.getenv("TTS_VOICE", "en-US-AriaNeural").strip()

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
            "API key is missing! Please set `LLM_API_KEY` in your `.env` file. "
            "You can use Groq (free at https://console.groq.com), OpenRouter, or run Ollama locally."
        )
    return True, ""
