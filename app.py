"""
app.py - Nova Health Streamlit User Interface

Main entry point for Nova Health.
Features:
1. Permanent clinical medical disclaimer.
2. Pre-LLM deterministic emergency & self-harm safety triage (safety.py).
3. Evidence-informed agentic ReAct health tools (health_tools.py).
4. Multi-user session profile isolation (user_id partitioning).
5. Interactive sidebar dashboard:
   - Metric trend line charts (st.line_chart)
   - Medication schedule with deletion
   - Symptom timeline formatted for physician visits
   - Health profile memory management
6. Voice narration (Edge-TTS) with hidden background audio player.
"""

import base64
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from config import MODEL_ID, TTS_VOICE, MEDICAL_DISCLAIMER, validate_config
from database import (
    init_db,
    load_messages,
    save_message,
    clear_chat_history,
    get_health_metrics,
    delete_health_metric,
    get_symptoms,
    delete_symptom,
    get_medication_reminders,
    delete_medication_reminder,
    get_preferences,
    delete_preference,
)
from safety import check_safety
from llm import stream_chat_response
from voice import speak

# ==============================================================================
# 1. Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Nova Health - Wellness & Lifestyle Assistant",
    page_icon="🌱",
    layout="wide",
)


def play_hidden_audio(mp3_bytes: bytes) -> None:
    """
    Plays MP3 audio silently in the background with zero visible UI footprint.
    Encodes raw audio into a Base64 data URI inside an invisible iframe (height=0).
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


# ==============================================================================
# 2. Database Initialization
# ==============================================================================
init_db()

# ==============================================================================
# 3. Sidebar Controls & Health Dashboard
# ==============================================================================
with st.sidebar:
    st.title("🌱 Nova Health")
    st.caption("Evidence-informed wellness assistant & lifestyle tracker.")
    st.markdown(f"**Chat Model:** `{MODEL_ID}`")

    # User Profile / Multi-Tenant Isolation
    user_id = st.text_input(
        "👤 Active Profile / Username",
        value="default_user",
        help="Health logs, symptoms, and reminders are partitioned by this username.",
    ).strip().lower() or "default_user"

    st.caption("🔒 *Privacy Note: Do not enter real personal health data in this shared demo.*")

    # Voice Narration Toggle
    speak_replies = st.toggle("🔊 Speak replies", value=False, help="Read answers aloud using Edge TTS")
    if speak_replies:
        st.caption(f"Voice: `{TTS_VOICE}`")

    # Clear chat button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        clear_chat_history(user_id)
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.subheader("📊 Health Dashboard")

    # --- Dashboard Section A: Health Metrics & Line Chart ---
    with st.expander("📈 Metrics & Trends", expanded=False):
        metrics_list = ["weight", "sleep_hours", "water_ml", "blood_sugar", "mood"]
        selected_metric = st.selectbox("Inspect Metric:", metrics_list)

        raw_metrics = get_health_metrics(user_id, metric=selected_metric, limit=15)
        if raw_metrics:
            # Build DataFrame for line chart (sorted chronologically)
            chart_df = pd.DataFrame(raw_metrics)[::-1]
            st.caption(f"Trend for **{selected_metric}**:")
            st.line_chart(chart_df.set_index("created_at")["value"], height=160)

            # Table of recent entries with delete buttons
            st.caption("Recent Entries:")
            for m in raw_metrics[:5]:
                col_info, col_btn = st.columns([4, 1])
                with col_info:
                    st.text(f"{m['created_at'][:10]}: {m['value']} {m['unit']}")
                with col_btn:
                    if st.button("❌", key=f"del_met_{m['id']}", help="Delete entry"):
                        delete_health_metric(m["id"])
                        st.rerun()
        else:
            st.caption(f"No entries logged for '{selected_metric}' yet. Ask Nova e.g. 'Log weight 70 kg'.")

    # --- Dashboard Section B: Medication Reminders ---
    with st.expander("💊 Medication Schedule", expanded=False):
        reminders = get_medication_reminders(user_id)
        if reminders:
            for r in reminders:
                col_info, col_btn = st.columns([4, 1])
                with col_info:
                    note = f" *({r['notes']})*" if r["notes"] else ""
                    st.markdown(f"**{r['name']}** at `{r['time']}`{note}")
                with col_btn:
                    if st.button("❌", key=f"del_med_{r['id']}", help="Delete reminder"):
                        delete_medication_reminder(r["id"])
                        st.rerun()
        else:
            st.caption("No medication reminders scheduled. Ask Nova e.g. 'Remind me to take Vitamin D at 8 AM'.")

    # --- Dashboard Section C: Symptom Timeline ---
    with st.expander("🩺 Symptom Log", expanded=False):
        symptoms = get_symptoms(user_id, limit=10)
        if symptoms:
            for s in symptoms:
                col_info, col_btn = st.columns([4, 1])
                sev = s["severity"]
                # Color-code severity tags
                badge = "🟢" if sev <= 3 else ("🟡" if sev <= 7 else "🔴")
                with col_info:
                    notes = f"\n*{s['notes']}*" if s["notes"] else ""
                    st.markdown(f"{badge} **{s['symptom']}** (Sev: {sev}/10) — `{s['created_at'][:10]}`{notes}")
                with col_btn:
                    if st.button("❌", key=f"del_symp_{s['id']}", help="Delete symptom"):
                        delete_symptom(s["id"])
                        st.rerun()
        else:
            st.caption("No symptoms recorded yet. Ask Nova e.g. 'Log mild knee pain severity 3'.")

    # --- Dashboard Section D: Health Profile & Memory ---
    with st.expander("🧠 Health Profile & Goals", expanded=False):
        prefs = get_preferences(user_id)
        if prefs:
            for k, v in prefs.items():
                col_info, col_btn = st.columns([4, 1])
                with col_info:
                    st.markdown(f"**{k}:** {v}")
                with col_btn:
                    if st.button("❌", key=f"del_pref_{k}", help="Delete preference"):
                        delete_preference(user_id, k)
                        st.rerun()
        else:
            st.caption("No profile preferences saved yet. Tell Nova e.g. 'I am vegetarian' or 'My goal is better sleep'.")

# ==============================================================================
# 4. Main Application Window & Disclaimer Banner
# ==============================================================================
st.title("🌱 Nova Health Virtual Assistant")
st.caption("Your companion for lifestyle tracking, nutrition exploration, and evidence-informed wellness habits.")

# Permanent Clinical Disclaimer Banner
st.info(f"⚕️ **Medical Disclaimer**: {MEDICAL_DISCLAIMER}")

# Configuration warning check
is_valid_config, config_error_msg = validate_config()
if not is_valid_config:
    st.warning(f"⚠️ **Setup Needed:** {config_error_msg}")

# Initialize user message history from SQLite
if "messages" not in st.session_state or st.session_state.get("active_user") != user_id:
    st.session_state.active_user = user_id
    st.session_state.messages = load_messages(user_id)

# Render previous chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==============================================================================
# 5. User Input & Execution Pipeline
# ==============================================================================
user_prompt = st.chat_input("Ask Nova Health about nutrition, metrics, BMI, or lifestyle habits...")

if user_prompt:
    # 1. Immediately render user message
    with st.chat_message("user"):
        st.markdown(user_prompt)

    save_message(user_id, "user", user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # 2. Pre-LLM Deterministic Safety Screening (Emergency & Crisis Interception)
    is_safe, safety_alert = check_safety(user_prompt)

    if not is_safe:
        # Bypasses the LLM completely: returns deterministic emergency/crisis advice
        with st.chat_message("assistant"):
            st.error(safety_alert)

        save_message(user_id, "assistant", safety_alert)
        st.session_state.messages.append({"role": "assistant", "content": safety_alert})

        # Voice Narration for safety alert if enabled
        if speak_replies:
            audio_bytes = speak(safety_alert)
            if audio_bytes:
                play_hidden_audio(audio_bytes)

    else:
        # 3. Safe to proceed with ReAct Agentic Execution Loop
        with st.chat_message("assistant"):
            tool_status_placeholder = st.empty()
            used_tools = []

            def on_tool_executed(tool_name: str) -> None:
                """Callback invoked whenever Nova Health executes an agentic health tool."""
                used_tools.append(tool_name)
                tool_status_placeholder.caption(f"🛠️ *Used tool:* `{tool_name}`")

            try:
                # Stream ONLY the final validated answer token-by-token
                full_response = st.write_stream(
                    stream_chat_response(
                        st.session_state.messages,
                        on_tool_call=on_tool_executed,
                        speak_enabled=speak_replies,
                        user_id=user_id,
                    )
                )

                # Persist tool usage badge above assistant response
                if used_tools:
                    tools_str = ", ".join(f"`{t}`" for t in used_tools)
                    tool_status_placeholder.caption(f"🛠️ *Used tool(s):* {tools_str}")

                # Save assistant response to SQLite and local state
                if full_response:
                    save_message(user_id, "assistant", full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                # Voice Narration: Speak final answer if toggle is enabled
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
