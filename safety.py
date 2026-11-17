"""
safety.py - Deterministic Safety & Clinical Red-Flag Screener

Acts as a pre-LLM deterministic firewall.
Screens user messages using keyword and regex matching BEFORE any AI inference occurs.
Intercepts acute medical emergencies, suicidal ideation/self-harm, and disordered
eating patterns, returning hardcoded, clinically safe responses.
"""

import re
from typing import Tuple
from config import EMERGENCY_NUMBER

# ==============================================================================
# 1. Acute Medical Emergency Patterns
# ==============================================================================
# Patterns for conditions requiring immediate emergency intervention (911/ER)
ACUTE_EMERGENCY_PATTERNS = [
    # Cardiovascular & Chest Pain
    r"\b(chest\s+pain|pain\s+in\s+(?:my\s+)?chest|pressure\s+in\s+(?:my\s+)?chest|heart\s+attack)\b",
    r"\b(pain\s+radiating\s+to\s+(?:my\s+)?(?:arm|jaw|neck))\b",
    # Respiratory Distress
    r"\b(can'?t\s+breathe|trouble\s+breathing|severe\s+shortness\s+of\s+breath|gasping\s+for\s+air|choking)\b",
    r"\b(difficulty\s+breathing\s+and\s+chest\s+tightness)\b",
    # Neurological & Stroke (FAST signs)
    r"\b(face\s+droop(?:ing)?|slurred\s+speech|slurring\s+(?:my\s+)?words|sudden\s+numbness|paralysis\s+on\s+one\s+side|stroke)\b",
    r"\b(sudden\s+loss\s+of\s+vision|sudden\s+inability\s+to\s+speak)\b",
    # Severe Bleeding & Trauma
    r"\b(severe\s+bleeding|bleeding\s+(?:heavily|uncontrollably)|coughing\s+up\s+blood|vomiting\s+blood)\b",
    # Poisoning & Overdose
    r"\b(overdose|took\s+too\s+many\s+pills|swallowed\s+poison|drank\s+bleach|poisoning)\b",
    # Anaphylaxis
    r"\b(throat\s+closing\s+up|swollen\s+tongue\s+and\s+can'?t\s+breathe|anaphylaxis)\b",
]

# ==============================================================================
# 2. Mental Health & Self-Harm Patterns
# ==============================================================================
# Patterns for psychological crises, self-injury, or suicidal ideation
SELF_HARM_PATTERNS = [
    r"\b(suicid(?:e|al)|kill\s+myself|end\s+my\s+life|want\s+to\s+die|better\s+off\s+dead)\b",
    r"\b(self[\s-]harm|cutting\s+myself|cut\s+my\s+wrists|hurting\s+myself)\b",
    r"\b(don'?t\s+want\s+to\s+live|thinking\s+of\s+ending\s+it)\b",
]

# ==============================================================================
# 3. Disordered Eating & Starvation Patterns
# ==============================================================================
# Patterns for compensatory behaviors, purging, or extreme starvation
DISORDERED_EATING_PATTERNS = [
    r"\b(purging|throw(?:ing)?\s+up\s+(?:after\s+eating|my\s+food)|make\s+myself\s+vomit)\b",
    r"\b(starv(?:ing|e)\s+myself|afraid\s+to\s+eat\s+anything|fear\s+of\s+eating|scared\s+of\s+calories)\b",
    r"\b(fast(?:ing)?\s+for\s+(?:5|6|7|8|9|10|\d{2,})\s+days)\b",
]


def check_safety(user_message: str) -> Tuple[bool, str]:
    """
    Evaluates an incoming user prompt for urgent clinical red flags before invoking the LLM.

    Args:
        user_message: The raw text string submitted by the user.

    Returns:
        Tuple[bool, str]:
          - is_safe (bool): True if message passed safety screening; False if a red flag matched.
          - response (str): Empty if safe; or a deterministic, hardcoded safety alert if flagged.
    """
    clean_text = user_message.strip().lower()
    if not clean_text:
        return True, ""

    # Check 1: Acute Medical Emergency (Physical Crisis)
    for pattern in ACUTE_EMERGENCY_PATTERNS:
        if re.search(pattern, clean_text, re.IGNORECASE):
            emergency_message = (
                f"🚨 **EMERGENCY MEDICAL ALERT**\n\n"
                f"Your message mentions symptoms that may indicate an **acute medical emergency** "
                f"(such as chest pain, severe breathing difficulty, stroke signs, or trauma).\n\n"
                f"⚠️ **Nova Health cannot provide emergency medical care or diagnosis.**\n\n"
                f"👉 **Please contact your local emergency services immediately by dialing {EMERGENCY_NUMBER} "
                f"or go to the nearest emergency department right now.** Do not delay medical attention."
            )
            return False, emergency_message

    # Check 2: Mental Health & Self-Harm (Crisis Intervention)
    for pattern in SELF_HARM_PATTERNS:
        if re.search(pattern, clean_text, re.IGNORECASE):
            crisis_message = (
                "💙 **You are not alone, and supportive help is available right now.**\n\n"
                "If you are experiencing thoughts of suicide, self-harm, or feeling overwhelmed, "
                "please connect with people who can support you through this moment:\n\n"
                "- **Suicide & Crisis Lifeline**: Call or text **988** (Free, confidential, 24/7 in the US & Canada)\n"
                "- **The Trevor Project** (LGBTQ youth): Call **1-866-488-7386** or text **START to 678-678**\n"
                "- **Crisis Text Line**: Text **HOME** to **741741**\n"
                "- **International Helplines**: Visit [findahelpline.com](https://findahelpline.com) for confidential support in your country.\n\n"
                "Please reach out to a trusted loved one, counselor, or doctor today. People care and want to help you."
            )
            return False, crisis_message

    # Check 3: Disordered Eating Guardrail
    for pattern in DISORDERED_EATING_PATTERNS:
        if re.search(pattern, clean_text, re.IGNORECASE):
            disorder_message = (
                "🌱 **A Note on Sustainable Well-being**\n\n"
                "Nova Health promotes safe, healthy, and nourishing relationships with food. "
                "Severe food restriction, purging, or deep distress surrounding eating can have serious "
                "physiological and emotional consequences.\n\n"
                "Rather than calculating restrictive diet targets, we warmly encourage discussing these feelings "
                "with a compassionate healthcare provider or specialized therapist.\n\n"
                "If you would like confidential support, you can also connect with the "
                "**National Eating Disorders Association (NEDA)** helpline or visit [neda.org](https://www.nationaleatingdisorders.org)."
            )
            return False, disorder_message

    # Passed all deterministic safety checks
    return True, ""
