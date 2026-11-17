"""
app.py - Streamlit User Interface for Nova Virtual Assistant

Main entry point for Nova.
Features:
1. Natural language streaming chat with persistent SQLite history.
2. PDF Document Upload & Q&A: Upload any PDF to ask questions, summarize, or extract details.
3. Image Upload & Vision: Upload images (JPG, PNG, WEBP) for Nova to describe, analyse, or answer questions about.
4. Autonomous ReAct Tool Calling: Weather, safe AST calculator, notes, reminders, user memory.
5. Voice Narration: Spoken replies using Edge-TTS with invisible HTML5 autoplay.
6. Interactive Sidebar: PDF/image manager, saved notes, task reminders, and long-term memory viewers.
"""

import os
import sys
from pathlib import Path

# Ensure application directory is first in sys.path to resolve local imports cleanly
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import base64
import io
import pypdf
import streamlit as st
import streamlit.components.v1 as components

from app_config import MODEL_ID, TTS_VOICE, validate_config
from database import (
    init_db,
    load_messages,
    save_message,
    clear_chat_history,
    get_preferences,
    delete_preference,
    get_notes,
    delete_note,
    get_reminders,
    delete_reminder,
)
from llm import stream_chat_response
from voice import speak

# 1. Page Configuration
st.set_page_config(
    page_title="Nova - Virtual Assistant",
    page_icon="✨",
    layout="centered",
)


def play_hidden_audio(mp3_bytes: bytes) -> None:
    """
    Plays MP3 audio silently in the background with zero visible UI footprint.
    Encodes the raw audio bytes into a Base64 data URI and renders an HTML5
    <audio autoplay> element inside an invisible iframe (height=0).
    """
    b64 = base64.b64encode(mp3_bytes).decode()
    components.html(
        f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """,
        height=0,
    )


def extract_text_from_pdf(file_bytes: bytes) -> tuple[str, int]:
    """
    Extracts plain text and page count from uploaded PDF bytes using pypdf.
    Adds page markers (--- Page X ---) so the LLM can reference specific pages.
    """
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    num_pages = len(reader.pages)
    pages_text = []

    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages_text.append(f"--- Page {idx + 1} ---\n{text.strip()}")

    full_text = "\n\n".join(pages_text)
    return full_text, num_pages


def encode_image_to_base64(file_bytes: bytes, mime_type: str) -> str:
    """
    Encodes raw image bytes into a Base64 data URI string.
    This is the format required by vision-capable LLM APIs to send inline images.
    Example output: 'data:image/jpeg;base64,/9j/4AAQSkZJRgAB...'
    """
    b64 = base64.b64encode(file_bytes).decode()
    return f"data:{mime_type};base64,{b64}"


# Map file extensions to their correct MIME types for the API
MIME_TYPE_MAP = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}


# 2. Database Initialization
init_db()

# 3. Sidebar Controls & Storage Viewers
with st.sidebar:
    st.title("✨ Nova Assistant")
    st.caption("A lightweight AI virtual assistant for your university lab.")
    st.markdown(f"**Chat Model:** `{MODEL_ID}`")

    # Voice Narration Toggle (default: Off)
    speak_replies = st.toggle(
        "🔊 Speak replies",
        value=False,
        help="Read assistant answers aloud using Edge TTS",
    )
    if speak_replies:
        st.caption(f"Voice: `{TTS_VOICE}`")

    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        clear_chat_history()
        st.session_state.messages = []
        st.session_state.pop("pdf_text", None)
        st.session_state.pop("pdf_name", None)
        st.session_state.pop("pdf_pages", None)
        st.session_state.pop("image_b64", None)
        st.session_state.pop("image_name", None)
        st.session_state.pop("image_mime", None)
        st.rerun()

    st.markdown("---")

    # -------------------------------------------------------------------------
    # PDF Document Upload Section
    # -------------------------------------------------------------------------
    st.subheader("📄 Document Q&A")
    uploaded_pdf = st.file_uploader(
        "Upload a PDF to ask questions:",
        type=["pdf"],
        help="Upload lecture notes, research papers, syllabus, or manuals.",
    )

    if uploaded_pdf is not None:
        # Only re-process if a new file was uploaded (avoids re-reading on every rerun)
        if (
            "pdf_name" not in st.session_state
            or st.session_state.pdf_name != uploaded_pdf.name
        ):
            try:
                with st.spinner("Extracting text from PDF..."):
                    pdf_bytes = uploaded_pdf.getvalue()
                    extracted_text, num_pages = extract_text_from_pdf(pdf_bytes)

                    if extracted_text.strip():
                        st.session_state.pdf_text = extracted_text
                        st.session_state.pdf_name = uploaded_pdf.name
                        st.session_state.pdf_pages = num_pages
                        st.success(f"Loaded {uploaded_pdf.name} ({num_pages} pages)")
                    else:
                        st.warning("⚠️ No readable text found in this PDF (it might be scanned images).")
            except Exception as err:
                st.error(f"Error reading PDF: {err}")

    # Display active PDF info and a remove button
    if "pdf_text" in st.session_state and st.session_state.pdf_text:
        st.info(f"📎 **Active Doc:** {st.session_state.pdf_name} ({st.session_state.pdf_pages} pages)")
        if st.button("❌ Remove PDF", use_container_width=True):
            st.session_state.pop("pdf_text", None)
            st.session_state.pop("pdf_name", None)
            st.session_state.pop("pdf_pages", None)
            st.rerun()

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Image Upload Section
    # -------------------------------------------------------------------------
    st.subheader("🖼️ Image Q&A")
    uploaded_image = st.file_uploader(
        "Upload an image to ask questions:",
        type=["jpg", "jpeg", "png", "gif", "webp"],
        help="Upload diagrams, charts, screenshots, or any photo for Nova to analyse.",
    )

    if uploaded_image is not None:
        # Only re-encode if a new image was uploaded
        if (
            "image_name" not in st.session_state
            or st.session_state.image_name != uploaded_image.name
        ):
            try:
                img_bytes = uploaded_image.getvalue()
                ext = uploaded_image.name.rsplit(".", 1)[-1].lower()
                mime = MIME_TYPE_MAP.get(ext, "image/jpeg")
                st.session_state.image_b64 = encode_image_to_base64(img_bytes, mime)
                st.session_state.image_name = uploaded_image.name
                st.session_state.image_mime = mime
                st.success(f"Loaded image: {uploaded_image.name}")
            except Exception as err:
                st.error(f"Error loading image: {err}")

    # Preview the active image and show a remove button
    if "image_b64" in st.session_state and st.session_state.image_b64:
        st.image(uploaded_image or st.session_state.image_b64, caption=st.session_state.image_name, use_container_width=True)
        if st.button("❌ Remove Image", use_container_width=True):
            st.session_state.pop("image_b64", None)
            st.session_state.pop("image_name", None)
            st.session_state.pop("image_mime", None)
            st.rerun()

    st.markdown("---")

    # Expandable viewer for saved user preferences (Long-Term Memory)
    with st.expander("🧠 User Memory", expanded=False):
        prefs = get_preferences()
        if not prefs:
            st.caption("No preferences saved yet. Tell Nova e.g. 'My name is Parth'.")
        else:
            for key, val in prefs.items():
                col_text, col_del = st.columns([4, 1])
                with col_text:
                    st.markdown(f"**{key}:** {val}")
                with col_del:
                    if st.button("❌", key=f"del_pref_{key}", help="Delete preference"):
                        delete_preference(key)
                        st.rerun()

    # Expandable viewer for saved notes
    with st.expander("📝 Saved Notes", expanded=False):
        notes = get_notes()
        if not notes:
            st.caption("No notes saved yet. Tell Nova e.g. 'Note: submit assignment'.")
        else:
            for note in notes:
                col_text, col_del = st.columns([4, 1])
                with col_text:
                    title_str = note["title"] or "Quick Note"
                    st.markdown(f"**{title_str}**\n{note['content']}")
                with col_del:
                    if st.button("❌", key=f"del_note_{note['id']}", help="Delete note"):
                        delete_note(note["id"])
                        st.rerun()

    # Expandable viewer for saved reminders
    with st.expander("⏰ Reminders", expanded=False):
        reminders = get_reminders()
        if not reminders:
            st.caption("No reminders set yet. Tell Nova e.g. 'Remind me to call John at 5pm'.")
        else:
            for rem in reminders:
                col_text, col_del = st.columns([4, 1])
                with col_text:
                    due = f" *({rem['due_time']})*" if rem["due_time"] else ""
                    st.markdown(f"• {rem['task']}{due}")
                with col_del:
                    if st.button("❌", key=f"del_rem_{rem['id']}", help="Delete reminder"):
                        delete_reminder(rem["id"])
                        st.rerun()

# 4. Configuration Check
is_valid_config, config_error_msg = validate_config()
if not is_valid_config:
    st.warning(f"⚠️ **Setup Needed:** {config_error_msg}")

# 5. Initialize Streamlit Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = load_messages()

# 6. Active Attachment Banners in Chat View
active_banners = []
if "pdf_text" in st.session_state and st.session_state.pdf_text:
    active_banners.append(f"📎 **PDF:** `{st.session_state.pdf_name}` ({st.session_state.pdf_pages} pages)")
if "image_b64" in st.session_state and st.session_state.image_b64:
    active_banners.append(f"🖼️ **Image:** `{st.session_state.image_name}`")
if active_banners:
    st.caption("  ·  ".join(active_banners) + " — Ask anything about your attachments!")

# 7. Render Saved Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 8. User Input Field
user_prompt = st.chat_input("Ask Nova something, about your PDF, or describe what you see in your image...")

# 9. Unified Chat & Execution Flow
if user_prompt:
    # Display the user's message immediately in the UI
    with st.chat_message("user"):
        st.markdown(user_prompt)
        # Show a small thumbnail preview below the message if an image is active
        if "image_b64" in st.session_state and st.session_state.image_b64:
            st.caption(f"🖼️ Attached: `{st.session_state.image_name}`")

    # Persist the plain text of the user message to SQLite and session state
    # (Images are passed fresh each time via the vision API; we don't store base64 in SQLite)
    save_message("user", user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Prepare PDF document context if one is currently uploaded (capped at ~60,000 chars for context safety)
    active_pdf_context = None
    active_pdf_name = None
    if "pdf_text" in st.session_state and st.session_state.pdf_text:
        raw_text = st.session_state.pdf_text
        if len(raw_text) > 60000:
            active_pdf_context = raw_text[:60000] + "\n\n[Note: Document truncated to first 60,000 characters.]"
        else:
            active_pdf_context = raw_text
        active_pdf_name = st.session_state.pdf_name

    # Prepare image context if an image is currently uploaded
    active_image_b64 = st.session_state.get("image_b64")
    active_image_name = st.session_state.get("image_name")

    # Generate assistant response with the agentic ReAct loop
    with st.chat_message("assistant"):
        tool_status_placeholder = st.empty()
        used_tools = []

        def on_tool_executed(tool_name: str) -> None:
            """Callback fired whenever Nova invokes a tool in the ReAct loop."""
            used_tools.append(tool_name)
            tool_status_placeholder.caption(f"🛠️ *Used tool:* `{tool_name}`")

        try:
            # Stream ONLY the final text response token-by-token
            full_response = st.write_stream(
                stream_chat_response(
                    st.session_state.messages,
                    on_tool_call=on_tool_executed,
                    speak_enabled=speak_replies,
                    pdf_context=active_pdf_context,
                    pdf_name=active_pdf_name,
                    image_b64=active_image_b64,
                    image_name=active_image_name,
                )
            )

            # Keep the final tool usage status badge visible
            if used_tools:
                tools_str = ", ".join(f"`{t}`" for t in used_tools)
                tool_status_placeholder.caption(f"🛠️ *Used tool:* {tools_str}")

            # Save the final assistant answer to database and session state
            if full_response:
                save_message("assistant", full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            # Voice Narration: If toggle is enabled, speak the answer
            if speak_replies and full_response:
                try:
                    audio_bytes = speak(full_response)
                    if audio_bytes:
                        play_hidden_audio(audio_bytes)
                    else:
                        st.warning("⚠️ Could not generate voice narration.")
                except Exception:
                    st.warning("⚠️ Voice narration unavailable.")

        except RuntimeError as err:
            st.error(f"⚠️ {err}")
        except Exception as exc:
            st.error(f"⚠️ An unexpected error occurred: {str(exc)}")
