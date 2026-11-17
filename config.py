"""
config.py - Backward Compatibility Wrapper

Re-exports configuration variables from app_config.py.
Prevents namespace collisions with third-party 'config' modules on Linux/Cloud environments.
"""

from app_config import (
    BASE_DIR,
    LLM_BASE_URL,
    LLM_API_KEY,
    MODEL_ID,
    EMERGENCY_NUMBER,
    TTS_VOICE,
    DB_PATH,
    MEDICAL_DISCLAIMER,
    SYSTEM_PROMPT,
    validate_config,
    get_setting,
)

__all__ = [
    "BASE_DIR",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "MODEL_ID",
    "EMERGENCY_NUMBER",
    "TTS_VOICE",
    "DB_PATH",
    "MEDICAL_DISCLAIMER",
    "SYSTEM_PROMPT",
    "validate_config",
    "get_setting",
]
