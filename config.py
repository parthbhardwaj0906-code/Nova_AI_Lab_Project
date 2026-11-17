"""
config.py - Alias for app_config.py

Provides backward compatibility for modules importing config directly.
"""

from app_config import (
    BASE_DIR,
    get_setting,
    LLM_BASE_URL,
    LLM_API_KEY,
    MODEL_ID,
    TTS_VOICE,
    DB_PATH,
    SYSTEM_PROMPT,
    validate_config,
)
