"""
llm.py - Universal LLM Interface with Tool Calling (ReAct Loop)

Features:
1. Strategy Pattern: Connects to any OpenAI-compatible provider (Groq, Ollama, OpenRouter, OpenAI).
2. Native Tool Calling: Sends tool schemas, runs requested functions, feeds results back (max 5 rounds).
3. Final Answer Streaming: Streams ONLY the assistant's final response tokens.
4. Tool Execution Callback: Fires an `on_tool_call` callback so app.py can display a UI badge.
5. Voice Mode Adaptation: Instructs model to keep responses brief when speech mode is active.
6. JSON Fallback: Automatically handles models lacking native function-calling support.
7. Long-term Memory: Injects user preferences from SQLite into the system prompt.
8. PDF Document Context: Seamlessly injects uploaded PDF content for Q&A and document analysis.
9. Image Vision: Sends base64-encoded images to vision-capable models for description and analysis.
"""

import json
from typing import Generator, List, Dict, Any, Callable, Optional
from openai import OpenAI
import openai

from app_config import LLM_BASE_URL, LLM_API_KEY, MODEL_ID, SYSTEM_PROMPT
from database import get_preferences
import tools

# Limit iterative tool-calling steps to prevent infinite execution loops
MAX_TOOL_ROUNDS = 5


def get_client() -> OpenAI:
    """
    Initializes and returns the OpenAI client with configured endpoint and key.
    Uses 'ollama' as dummy key for local endpoints if no key is provided.
    """
    api_key = LLM_API_KEY or "ollama"
    return OpenAI(base_url=LLM_BASE_URL, api_key=api_key)


def build_system_prompt(
    speak_enabled: bool = False,
    pdf_context: Optional[str] = None,
    pdf_name: Optional[str] = None,
    image_b64: Optional[str] = None,
    image_name: Optional[str] = None,
) -> str:
    """
    Prepares the full system prompt by combining Nova's personality with
    persisted user preferences or facts retrieved from SQLite, active PDF context,
    active image context hint, and voice narration instructions.
    """
    prompt = SYSTEM_PROMPT

    try:
        prefs = get_preferences()
        if prefs:
            pref_lines = [f"- {k}: {v}" for k, v in prefs.items()]
            prompt += (
                "\n\nUser Profile & Saved Long-Term Memory:\n"
                + "\n".join(pref_lines)
                + "\nAlways personalize answers based on these saved preferences."
            )
        prompt += (
            "\n\nMemory Instructions: When the user tells you their name, preferences, "
            "or background (e.g., 'My name is X', 'I prefer short answers'), call the "
            "`remember_preference` tool to permanently save it."
        )
    except Exception:
        pass

    # If an uploaded PDF document is active, inject it into the system prompt
    if pdf_context:
        doc_label = pdf_name or "Uploaded Document"
        prompt += (
            f"\n\nActive Uploaded Document: '{doc_label}'\n"
            "The user has uploaded this PDF document for you to reference in this conversation. "
            "Use its contents to answer questions, explain concepts, provide summaries, or extract details:\n"
            "<DOCUMENT_CONTENT>\n"
            f"{pdf_context}\n"
            "</DOCUMENT_CONTENT>\n"
            "When answering questions about this document, cite relevant page numbers when available."
        )

    # If an image is active, add a brief hint (the actual image is sent in the user message content)
    if image_b64:
        img_label = image_name or "uploaded image"
        prompt += (
            f"\n\nActive Uploaded Image: '{img_label}'\n"
            "The user has attached an image to this conversation. "
            "The image is included directly in the user's message. "
            "Carefully examine it and use it to answer questions, describe what you see, or extract information."
        )

    # Voice instruction when audio narration is enabled
    if speak_enabled:
        prompt += "\n\nVoice Narration Active: Keep answers brief and conversational, since they will be spoken aloud."

    return prompt


def _stream_native_tools(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    on_tool_call: Optional[Callable[[str], None]] = None,
) -> Generator[str, None, None]:
    """
    Executes the agentic tool-use loop using OpenAI's native tool calling protocol.
    Repeats up to MAX_TOOL_ROUNDS:
      - If tool calls are requested: executes them, calls on_tool_call, and feeds observations back.
      - If no tool calls are requested: streams final answer tokens to the generator.
    """
    current_messages = list(messages)

    for round_idx in range(MAX_TOOL_ROUNDS):
        # Request completion with tools enabled
        response_stream = client.chat.completions.create(
            model=MODEL_ID,
            messages=current_messages,
            tools=tools.TOOLS_SCHEMA,
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

        # If tools were requested, execute each one and append the results
        tool_calls_list = []
        for idx in sorted(tool_calls_buffer.keys()):
            tc_data = tool_calls_buffer[idx]
            tool_calls_list.append({
                "id": tc_data["id"],
                "type": "function",
                "function": {
                    "name": tc_data["name"],
                    "arguments": tc_data["arguments"],
                },
            })

        # Append assistant's tool-call request to the conversation history
        current_messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls_list,
        })

        # Execute each requested tool and add its observation
        for tc_data in tool_calls_list:
            tool_name = tc_data["function"]["name"]
            raw_args = tc_data["function"]["arguments"]

            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                args = {}

            if on_tool_call:
                on_tool_call(tool_name)

            observation = tools.execute_tool(tool_name, args)

            current_messages.append({
                "role": "tool",
                "tool_call_id": tc_data["id"],
                "content": str(observation),
            })


def _stream_json_fallback(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    on_tool_call: Optional[Callable[[str], None]] = None,
) -> Generator[str, None, None]:
    """
    Fallback agentic loop for models that do not natively support function calling.
    Instructs the model to output a structured JSON block when it wants to call a tool.
    """
    tools_prompt = (
        "\n\nYou have access to these tools:\n"
        + json.dumps(tools.TOOLS_SCHEMA, indent=2)
        + "\n\nTo use a tool, respond ONLY with a JSON object: "
        '{"tool": "<name>", "arguments": {<args>}}.\n'
        "If no tool is required, respond with your normal plain text answer."
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

        # Check if the model decided to call a tool via JSON
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

            observation = tools.execute_tool(tool_name, tool_args)

            # Feed the observation back to the model for the next turn
            current_messages.append({"role": "assistant", "content": reply_text})
            current_messages.append({
                "role": "user",
                "content": f"Observation for tool '{tool_name}': {observation}",
            })
        else:
            # Model generated the final plain-text answer: stream it token-by-token
            for token in reply_text:
                yield token
            break


def stream_chat_response(
    messages: List[Dict[str, str]],
    on_tool_call: Optional[Callable[[str], None]] = None,
    speak_enabled: bool = False,
    pdf_context: Optional[str] = None,
    pdf_name: Optional[str] = None,
    image_b64: Optional[str] = None,
    image_name: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Main entry point invoked by app.py.
    Orchestrates communication with the LLM, managing tool calling rounds
    and streaming only the final answer.

    Args:
        messages: List of chat messages with 'role' and 'content'.
        on_tool_call: Optional callback invoked with the tool name whenever a tool runs.
        speak_enabled: Whether voice narration is enabled (adapts system prompt).
        pdf_context: Optional extracted text from an uploaded PDF.
        pdf_name: Optional filename of the uploaded PDF.
        image_b64: Optional base64 data URI of an uploaded image (e.g. 'data:image/jpeg;base64,...').
        image_name: Optional filename of the uploaded image (used in the system prompt hint).

    Yields:
        str: Tokens representing the assistant's final textual answer.
    """
    # 1. Validation check
    is_local = "localhost" in LLM_BASE_URL or "127.0.0.1" in LLM_BASE_URL
    if not is_local and (not LLM_API_KEY or "your_" in LLM_API_KEY):
        raise RuntimeError(
            "API key is missing! Please configure `LLM_API_KEY` in your `.env` file or Streamlit Secrets. "
            "Get a free key from Groq (https://console.groq.com) or run Ollama locally."
        )

    # 2. Build system prompt with memory, PDF context, and image hint
    sys_prompt = build_system_prompt(
        speak_enabled=speak_enabled,
        pdf_context=pdf_context,
        pdf_name=pdf_name,
        image_b64=image_b64,
        image_name=image_name,
    )
    full_messages = [{"role": "system", "content": sys_prompt}] + list(messages)

    # 3. If an image is active, upgrade the latest user message to a multimodal content block.
    #    The OpenAI vision format requires the content field to be a list of parts:
    #    [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "data:..."}}]
    if image_b64 and full_messages:
        # Find the last user message in the conversation and add the image to it
        for i in range(len(full_messages) - 1, -1, -1):
            if full_messages[i]["role"] == "user":
                user_text = full_messages[i].get("content", "")
                full_messages[i] = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": image_b64}},
                    ],
                }
                break

    client = get_client()

    try:
        # First attempt native OpenAI function calling
        yield from _stream_native_tools(client, full_messages, on_tool_call=on_tool_call)

    except openai.BadRequestError as bad_req:
        err_msg = str(bad_req).lower()
        # Fall back to JSON mode if provider complains that tools are unsupported
        if "tool" in err_msg or "support" in err_msg or "parameter" in err_msg:
            try:
                yield from _stream_json_fallback(
                    client, full_messages, on_tool_call=on_tool_call
                )
                return
            except Exception as fb_err:
                raise RuntimeError(f"Fallback error: {fb_err}") from fb_err
        raise RuntimeError(f"API Bad Request (400): {bad_req}") from bad_req

    except openai.AuthenticationError as auth_err:
        raise RuntimeError(
            "Invalid API Key (401 Unauthorized). "
            "Please check `LLM_API_KEY` in your `.env` file or Streamlit Secrets."
        ) from auth_err

    except (openai.PermissionDeniedError, openai.RateLimitError) as rate_err:
        raise RuntimeError(
            "Out of credits or rate limited, try another provider in .env."
        ) from rate_err

    except openai.NotFoundError as not_found:
        raise RuntimeError(
            f"Model or endpoint not found (404) at {LLM_BASE_URL}. "
            f"Verify `MODEL_ID` ('{MODEL_ID}') in `.env`."
        ) from not_found

    except openai.APIConnectionError as conn_err:
        raise RuntimeError(
            f"Could not connect to {LLM_BASE_URL}. "
            "If using local Ollama, ensure the Ollama server is running."
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
