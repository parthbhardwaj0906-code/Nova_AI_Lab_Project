"""
app.py - Streamlit User Interface for Nova

Entry point for the Nova Virtual Assistant.
Renders the chat interface, manages conversation history,
supports spoken replies (Edge-TTS) via a sidebar toggle,
displays real-time "Used tool: X" badges, streams final answers,
and provides a sidebar to view and delete long-term memories, notes, and reminders.
"""

import base64
import streamlit as st
import streamlit.components.v1 as components
from config import MODEL_ID, TTS_VOICE, validate_config
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


# 2. Database Initialization
# Ensures tables (messages, notes, reminders, user_preferences) exist
init_db()

# 4. Sidebar Controls & Storage Viewers
with st.sidebar:
    st.title("✨ Nova Assistant")
    st.caption("A lightweight AI virtual assistant for your university lab.")
    st.markdown(f"**Chat Model:** `{MODEL_ID}`")

    # Voice Narration Toggle (default: Off)
    speak_replies = st.toggle("🔊 Speak replies", value=False, help="Read assistant answers aloud using Edge TTS")
    if speak_replies:
        st.caption(f"Voice: `{TTS_VOICE}`")

    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        clear_chat_history()
        st.session_state.messages = []
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
            st.caption("No reminders set yet.")
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

    st.markdown("---")
    st.markdown("### 🧪 Quick Test Prompts")
    st.caption("Click any prompt below to test Nova's tools & memory:")

    if st.button("🌤️ What's the weather in Rajkot?", use_container_width=True):
        st.session_state.pending_prompt = "What's the weather in Rajkot?"
    if st.button("📝 Note: submit lab report Friday", use_container_width=True):
        st.session_state.pending_prompt = "Note: submit lab report Friday"
    if st.button("🔢 What is 15% of 2400?", use_container_width=True):
        st.session_state.pending_prompt = "What is 15% of 2400?"
    if st.button("🧠 My name is Parth and I study CS", use_container_width=True):
        st.session_state.pending_prompt = "My name is Parth and I study CS. Please remember that."

# 4. Configuration Check
is_valid_config, config_error_msg = validate_config()
if not is_valid_config:
    st.warning(f"⚠️ **Setup Needed:** {config_error_msg}")

# 5. Initialize Streamlit Session State
if "messages" not in st.session_state:
    st.session_state.messages = load_messages()

# 6. Render Saved Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 7. Determine Active User Input (Typed Text or Clicked Quick Prompt)
user_prompt = st.chat_input("Ask Nova something...")

if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
    user_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

# 8. Unified Chat & Execution Flow
if user_prompt:
    # Display the user's message immediately in the UI
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Persist user message to SQLite and session state
    save_message("user", user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Generate assistant response with the agentic ReAct loop
    with st.chat_message("assistant"):
        # Placeholder to dynamically display "Used tool: X" indicator
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
                )
            )

            # Keep the final tool usage status badge visible above or beside the answer
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
