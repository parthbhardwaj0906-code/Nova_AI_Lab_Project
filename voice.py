"""
voice.py - Text-to-Speech (TTS) Voice Narration for Nova

Uses Microsoft Edge's free neural TTS service via edge-tts.
Prepares and cleans assistant messages (stripping markdown syntax, code, tables)
and generates MP3 audio bytes for playback in the Streamlit UI.
"""

import asyncio
import concurrent.futures
import re
from typing import Optional
import edge_tts
from config import TTS_VOICE

# Maximum characters to speak before truncating and asking the user to read the screen
MAX_SPEECH_LENGTH = 600


def clean_text_for_speech(text: str) -> str:
    """
    Cleans markdown formatting out of assistant responses so the TTS engine
    sounds natural and does not read out code, symbols, or formatting tags.

    Steps:
    1. Removes multi-line code blocks (``` ... ```) and inline code (`...`).
    2. Drops Markdown tables (lines starting and ending with |).
    3. Converts markdown links [link text](url) to just 'link text'.
    4. Strips headers (#), bold/italics (*, _), and bullet indicators.
    5. Truncates text exceeding ~600 characters with an audio notice.
    """
    if not text:
        return ""

    # 1. Strip multi-line code blocks entirely (code is meant to be read, not spoken)
    cleaned = re.sub(r"```[\s\S]*?```", "", text)

    # 2. Convert inline backticks `code` into plain text
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)

    # 3. Filter out markdown table rows
    lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            continue
        lines.append(line)
    cleaned = "\n".join(lines)

    # 4. Turn markdown links [label](url) into just the label
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)

    # 5. Remove header markers (#, ##, etc.)
    cleaned = re.sub(r"#+\s*", "", cleaned)

    # 6. Remove bold, italic, and strikethrough characters (*, _, ~)
    cleaned = re.sub(r"[*_~]", "", cleaned)

    # 7. Remove blockquote markers (>) and bullet dashes (-)
    cleaned = re.sub(r"^\s*>\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)

    # 8. Collapse redundant whitespace and newlines into single spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # 9. Cap speech length to ~600 characters to keep audio responses punchy
    if len(cleaned) > MAX_SPEECH_LENGTH:
        # Cut at the nearest sentence end before the limit if possible
        sentence_end = cleaned[:MAX_SPEECH_LENGTH].rfind(".")
        if sentence_end > 350:
            cleaned = cleaned[: sentence_end + 1] + " See the screen for the rest."
        else:
            cleaned = cleaned[:MAX_SPEECH_LENGTH].rsplit(" ", 1)[0] + "... See the screen for the rest."

    return cleaned


async def _synthesize_async(text: str, voice: str) -> bytes:
    """
    Asynchronously streams speech audio chunks from the Microsoft Edge TTS service
    and aggregates them into a single MP3 byte buffer.
    """
    communicate = edge_tts.Communicate(text, voice)
    audio_chunks = []

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])

    return b"".join(audio_chunks)


def _run_async_safely(coro) -> bytes:
    """
    Executes an asynchronous coroutine synchronously, safely handling
    Streamlit threads that might already have an active event loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # If an event loop is already active in this thread, execute in a separate worker
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


def speak(text: str) -> Optional[bytes]:
    """
    Main entry point for text-to-speech narration.
    Cleans text, synthesizes audio via edge-tts, and returns MP3 bytes.

    Args:
        text: The assistant's final response markdown string.

    Returns:
        Optional[bytes]: Raw MP3 audio bytes, or None if synthesis fails.
    """
    speech_text = clean_text_for_speech(text)
    if not speech_text:
        return None

    try:
        # Generate MP3 audio using the configured voice (default: en-US-AriaNeural)
        audio_bytes = _run_async_safely(_synthesize_async(speech_text, TTS_VOICE))
        return audio_bytes if audio_bytes else None
    except Exception:
        # Return None gracefully on network timeout or connection failure (never crash)
        return None
