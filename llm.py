"""
llm.py - Universal LLM Interface for Nova Health (ReAct Loop)

Features:
1. Strategy Pattern: Connects to any OpenAI-compatible provider (Groq, Ollama, OpenRouter).
2. Clinical Safety Guardrails: Enforces evidence-informed wellness boundaries.
3. Native Tool Calling: Orchestrates health_tools (BMI, calories, metrics, nutrition, symptoms).
4. Final Answer Streaming: Streams ONLY the assistant's final response tokens.
5. User Profile Partitioning: Passes user_id for multi-tenant database isolation.
6. JSON Fallback: Automatically handles models lacking native function-calling support.
"""

import json
from typing import Generator, List, Dict, Any, Callable, Optional
from openai import OpenAI
import openai

from app_config import LLM_BASE_URL, LLM_API_KEY, MODEL_ID, SYSTEM_PROMPT
from database import get_preferences
import health_tools

# Limit iterative tool-calling steps to prevent infinite execution loops
MAX_TOOL_ROUNDS = 5


def get_client() -> OpenAI:
    """
    Initializes and returns the OpenAI client with configured endpoint and key.
    Uses 'ollama' as dummy key for local endpoints if no key is provided.
    """
    api_key = LLM_API_KEY or "ollama"
    return OpenAI(base_url=LLM_BASE_URL, api_key=api_key)


def build_system_prompt(user_id: str = "default_user", speak_enabled: bool = False) -> str:
    """
    Prepares the full system prompt by combining Nova Health's clinical boundaries
    with user-specific health profile preferences retrieved from SQLite.
    """
    prompt = SYSTEM_PROMPT

    try:
        prefs = get_preferences(user_id)
        if prefs:
            pref_lines = [f"- {k}: {v}" for k, v in prefs.items()]
            prompt += (
                f"\n\nActive User Profile ({user_id}) & Health Preferences:\n"
                + "\n".join(pref_lines)
                + "\nPersonalize lifestyle guidance based on these saved preferences when helpful."
            )
        prompt += (
            "\n\nTool Use Guidance: When the user shares health metrics (weight, water, sleep), "
            "asks for nutrition lookups, or reports symptoms, use the appropriate health tools to log "
            "or calculate data. Always remind users that you provide wellness support, not medical diagnoses."
        )
    except Exception:
        pass

    # Voice instruction when audio narration is enabled
    if speak_enabled:
        prompt += "\n\nVoice Narration Active: Keep answers brief, conversational, and easy to follow when spoken aloud."

    return prompt


def _stream_native_tools(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    user_id: str = "default_user",
    on_tool_call: Optional[Callable[[str], None]] = None,
) -> Generator[str, None, None]:
    """
    Executes the agentic tool-use loop using OpenAI's native tool calling protocol.
    Repeats up to MAX_TOOL_ROUNDS:
      - If tool calls are requested: executes health tools, calls on_tool_call, and feeds observations back.
      - If no tool calls are requested: streams final answer tokens to the generator.
    """
    current_messages = list(messages)

    for round_idx in range(MAX_TOOL_ROUNDS):
        response_stream = client.chat.completions.create(
            model=MODEL_ID,
            messages=current_messages,
            tools=health_tools.HEALTH_TOOLS_SCHEMA,
            tool_choice="auto",
            stream=True,
            temperature=0.7,
        )

        tool_calls_buffer: Dict[int, Dict[str, Any]] = {}

        for chunk in response_stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            # Case A: Model is streaming tool call fragments
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {
                            "id": tc.id or "",
                            "name": "",
                            "arguments": "",
                        }
                    if tc.id:
                        tool_calls_buffer[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_buffer[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_calls_buffer[idx]["arguments"] += tc.function.arguments

            # Case B: Model is streaming the final answer text (no tool calls this turn)
            elif delta.content and not tool_calls_buffer:
                yield delta.content

        # If no tools were invoked, this round was the final answer; we are done
        if not tool_calls_buffer:
            break

        # If tools WERE invoked, execute them and continue the ReAct loop
        assistant_tool_calls = []
        tool_results = []

        for tc_data in tool_calls_buffer.values():
            tool_id = tc_data["id"] or f"call_{round_idx}"
            tool_name = tc_data["name"]
            raw_args = tc_data["arguments"]

            try:
                parsed_args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                parsed_args = {}

            # Notify the UI that a tool was used (renders "Used tool: X")
            if on_tool_call:
                on_tool_call(tool_name)

            # Execute the health tool safely via health_tools.py
            observation = health_tools.execute_health_tool(tool_name, parsed_args, user_id=user_id)

            assistant_tool_calls.append({
                "id": tool_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": raw_args,
                },
            })

            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": str(observation),
            })

        # Append assistant tool-call request and tool observations to message history
        current_messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_tool_calls,
        })
        current_messages.extend(tool_results)


def _stream_json_fallback(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    user_id: str = "default_user",
    on_tool_call: Optional[Callable[[str], None]] = None,
) -> Generator[str, None, None]:
    """
    Fallback mechanism for models that do not natively support tool calling.
    Instructs the model to output a structured JSON object when it wants to call a health tool.
    Streams only the final textual answer.
    """
    tools_prompt = (
        "\n\nYou have access to the following health tools:\n"
        "1. calculate_bmi(weight_kg, height_cm) - BMI & WHO category.\n"
        "2. estimate_daily_calories(age, sex, weight_kg, height_cm, activity_level, goal) - Mifflin-St Jeor with safe floors.\n"
        "3. log_health_metric(metric, value, unit) - Log biometric data (weight, sleep, water, blood_sugar, blood_pressure, mood).\n"
        "4. get_health_history(metric, days) - View recent entries and trend.\n"
        "5. add_medication_reminder(name, time, notes) - Save medication schedule.\n"
        "6. list_medication_reminders() - View active medication reminders.\n"
        "7. log_symptom(symptom, severity_1_to_10, notes) - Record symptom with severity.\n"
        "8. get_symptom_history(days) - View symptom timeline for doctor visit.\n"
        "9. lookup_nutrition(food) - Query Open Food Facts for calories and macros.\n"
        "10. remember_health_preference(key, value) - Save health profile traits.\n\n"
        "If you need to call a tool, reply ONLY with a valid JSON object in this format:\n"
        "{\"tool\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n"
        "Do not include any conversational filler when calling a tool.\n"
        "If no tool is needed, respond directly with your final answer."
    )

    current_messages = list(messages)
    if current_messages and current_messages[0]["role"] == "system":
        current_messages[0] = {
            "role": "system",
            "content": current_messages[0]["content"] + tools_prompt,
        }

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=current_messages,
            stream=False,
            temperature=0.7,
        )

        reply_text = (response.choices[0].message.content or "").strip()

        is_tool_call = False
        parsed_call = None

        if reply_text.startswith("{") and reply_text.endswith("}"):
            try:
                data = json.loads(reply_text)
                if "tool" in data:
                    is_tool_call = True
                    parsed_call = data
            except Exception:
                pass

        if is_tool_call and parsed_call:
            tool_name = parsed_call.get("tool", "")
            tool_args = parsed_call.get("arguments", {})

            if on_tool_call:
                on_tool_call(tool_name)

            observation = health_tools.execute_health_tool(tool_name, tool_args, user_id=user_id)

            current_messages.append({"role": "assistant", "content": reply_text})
            current_messages.append({
                "role": "user",
                "content": f"Observation for tool '{tool_name}': {observation}",
            })
        else:
            for token in reply_text:
                yield token
            break


def stream_chat_response(
    messages: List[Dict[str, str]],
    on_tool_call: Optional[Callable[[str], None]] = None,
    speak_enabled: bool = False,
    user_id: str = "default_user",
) -> Generator[str, None, None]:
    """
    Main entry point invoked by app.py.
    Orchestrates communication with the LLM, managing health tool calling rounds
    and streaming only the final answer.

    Args:
        messages: List of chat messages with 'role' and 'content'.
        on_tool_call: Optional callback invoked with the tool name whenever a tool runs.
        speak_enabled: Whether voice narration is enabled (adapts system prompt).
        user_id: The active user profile identifier for isolated data access.

    Yields:
        str: Tokens representing the assistant's final textual answer.
    """
    is_local = "localhost" in LLM_BASE_URL or "127.0.0.1" in LLM_BASE_URL
    if not is_local and (not LLM_API_KEY or "your_" in LLM_API_KEY):
        raise RuntimeError(
            "API key is missing! Please configure `LLM_API_KEY` in your `.env` file. "
            "Get a free key from Groq (https://console.groq.com) or run Ollama locally."
        )

    # Build conversation history with system prompt & memory
    full_messages = [
        {"role": "system", "content": build_system_prompt(user_id=user_id, speak_enabled=speak_enabled)}
    ] + messages

    client = get_client()

    try:
        yield from _stream_native_tools(
            client, full_messages, user_id=user_id, on_tool_call=on_tool_call
        )

    except openai.BadRequestError as bad_req:
        err_msg = str(bad_req).lower()
        if "tool" in err_msg or "support" in err_msg or "parameter" in err_msg:
            try:
                yield from _stream_json_fallback(
                    client, full_messages, user_id=user_id, on_tool_call=on_tool_call
                )
                return
            except Exception as fb_err:
                raise RuntimeError(f"Fallback error: {fb_err}") from fb_err
        raise RuntimeError(f"API Bad Request (400): {bad_req}") from bad_req

    except openai.AuthenticationError as auth_err:
        raise RuntimeError(
            "Invalid API Key (401 Unauthorized). Please check `LLM_API_KEY` in your `.env` file."
        ) from auth_err

    except (openai.PermissionDeniedError, openai.RateLimitError) as rate_err:
        raise RuntimeError(
            "Out of credits or rate limited, try another provider in .env."
        ) from rate_err

    except openai.NotFoundError as not_found:
        raise RuntimeError(
            f"Model or endpoint not found (404) at {LLM_BASE_URL}. Verify `MODEL_ID` in `.env`."
        ) from not_found

    except openai.APIConnectionError as conn_err:
        raise RuntimeError(
            f"Could not connect to {LLM_BASE_URL}. If using local Ollama, ensure it is running."
        ) from conn_err

    except openai.APIStatusError as status_err:
        if status_err.status_code in (402, 429):
            raise RuntimeError(
                "Out of credits or rate limited, try another provider in .env."
            ) from status_err
        raise RuntimeError(
            f"API error ({status_err.status_code}): {status_err.message}"
        ) from status_err

    except Exception as exc:
        raise RuntimeError(f"Unexpected error: {str(exc)}") from exc
