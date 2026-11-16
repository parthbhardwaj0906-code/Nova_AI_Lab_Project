"""
tools.py - Nova Tool Suite

Provides 6 standalone, safe tools:
1. get_weather: Real-time weather via Open-Meteo REST API (free, no API key needed).
2. calculator: Safe AST-based mathematical expression evaluator (no dangerous eval()).
3. add_note: Saves personal notes to the SQLite database.
4. list_notes: Retrieves saved personal notes from the SQLite database.
5. add_reminder: Saves tasks and reminders to the SQLite database.
6. remember_preference: Saves long-term user facts and preferences to SQLite.

All tools return human-readable strings to be observed by the LLM.
"""

import ast
import math
import operator
import re
from typing import Dict, Any, List
import requests
from database import (
    save_note,
    get_notes,
    save_reminder,
    get_reminders,
    set_preference,
)


# ==============================================================================
# 1. Weather Tool (Open-Meteo)
# ==============================================================================
WEATHER_CODE_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
}


def get_weather(city: str) -> str:
    """
    Fetches real-time weather information for a given city using Open-Meteo API.
    Does not require any API keys.
    """
    try:
        clean_city = city.strip()
        # Step 1: Geocode the city name to latitude and longitude coordinates
        geo_url = (
            f"https://geocoding-api.open-meteo.com/v1/search?"
            f"name={clean_city}&count=1&language=en&format=json"
        )
        geo_res = requests.get(geo_url, timeout=5).json()

        if not geo_res.get("results"):
            return f"Could not find coordinates for city: '{clean_city}'."

        location = geo_res["results"][0]
        lat, lon = location["latitude"], location["longitude"]
        city_name = location.get("name", clean_city)
        country = location.get("country", "")

        # Step 2: Fetch current weather for the resolved coordinates
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
        )
        weather_res = requests.get(weather_url, timeout=5).json()
        current = weather_res.get("current", {})

        temp = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        wind = current.get("wind_speed_10m")
        code = current.get("weather_code", 0)
        condition = WEATHER_CODE_MAP.get(code, "Clear")

        return (
            f"Current weather in {city_name}, {country}: {condition}, "
            f"Temperature: {temp}°C, Humidity: {humidity}%, Wind: {wind} km/h."
        )
    except Exception as e:
        return f"Error fetching weather for '{city}': {str(e)}"


# ==============================================================================
# 2. Safe Calculator (AST Parser - No eval())
# ==============================================================================
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "abs": abs,
    "round": round,
    "pi": math.pi,
    "e": math.e,
}


def _eval_ast_node(node: ast.AST) -> Any:
    """
    Recursively walks and evaluates mathematical AST nodes safely.
    Rejects any unapproved nodes (such as imports, system calls, or assignments).
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric literals are allowed in calculations.")

    if isinstance(node, ast.BinOp):
        left_val = _eval_ast_node(node.left)
        right_val = _eval_ast_node(node.right)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op_type](left_val, right_val)
        raise ValueError(f"Operator {op_type.__name__} is not supported.")

    if isinstance(node, ast.UnaryOp):
        operand_val = _eval_ast_node(node.operand)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op_type](operand_val)
        raise ValueError(f"Unary operator {op_type.__name__} is not supported.")

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCTIONS:
            func = ALLOWED_FUNCTIONS[node.func.id]
            args = [_eval_ast_node(arg) for arg in node.args]
            return func(*args)
        raise ValueError("Unsupported mathematical function call.")

    if isinstance(node, ast.Name):
        if node.id in ALLOWED_FUNCTIONS:
            return ALLOWED_FUNCTIONS[node.id]
        raise ValueError(f"Unknown symbol: {node.id}")

    raise ValueError(f"Unsupported expression syntax: {type(node).__name__}")


def calculator(expression: str) -> str:
    """
    Safely calculates mathematical expressions without using python eval().
    Supports standard arithmetic, powers, roots, and percentages (e.g. '15% of 2400').
    """
    try:
        expr = expression.strip()

        # Pre-process percentage expressions like "15% of 2400" -> "(15 / 100.0) * 2400"
        expr = re.sub(
            r"(\d+(?:\.\d+)?)\s*%\s*(?:of\s*)?(\d+(?:\.\d+)?)",
            r"(\1 / 100.0) * \2",
            expr,
            flags=re.IGNORECASE,
        )
        expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1 / 100.0)", expr)

        # Parse the expression into an Abstract Syntax Tree (AST)
        parsed_ast = ast.parse(expr, mode="eval")
        result = _eval_ast_node(parsed_ast.body)

        # Format floating points cleanly (e.g., 360.0 -> 360)
        if isinstance(result, float) and result.is_integer():
            result = int(result)

        return f"Result: {result}"
    except Exception as err:
        return f"Calculation error: {str(err)}"


# ==============================================================================
# 3. Notes Tools (SQLite)
# ==============================================================================
def add_note(title: str = "", content: str = "") -> str:
    """Saves a note to the SQLite database."""
    try:
        if not content and title:
            content = title
            title = "Quick Note"
        elif not title and content:
            title = "Quick Note"
        elif not content and not title:
            return "Failed to save note: Content cannot be empty."

        note_id = save_note(title.strip(), content.strip())
        return f"Note #{note_id} ('{title}') successfully saved."
    except Exception as e:
        return f"Failed to save note: {str(e)}"


def list_notes() -> str:
    """Retrieves all notes saved in the SQLite database."""
    try:
        notes = get_notes()
        if not notes:
            return "No notes found in your database."
        lines = [
            f"- [#{n['id']}] {n['title'] or 'Untitled'}: {n['content']} (saved: {n['created_at']})"
            for n in notes
        ]
        return "Saved Notes:\n" + "\n".join(lines)
    except Exception as e:
        return f"Failed to list notes: {str(e)}"


# ==============================================================================
# 4. Reminder Tool (SQLite)
# ==============================================================================
def add_reminder(task: str, due_time: str = "") -> str:
    """Saves a task or reminder to the SQLite database."""
    try:
        clean_task = task.strip()
        if not clean_task:
            return "Failed to add reminder: Task description cannot be empty."

        rem_id = save_reminder(clean_task, due_time.strip())
        time_suffix = f" for '{due_time}'" if due_time else ""
        return f"Reminder #{rem_id} ('{clean_task}'){time_suffix} successfully created."
    except Exception as e:
        return f"Failed to add reminder: {str(e)}"


# ==============================================================================
# 5. Long-Term Memory / Preferences Tool (SQLite)
# ==============================================================================
def remember_preference(key: str, value: str) -> str:
    """
    Saves a user preference or personal detail into long-term memory.
    Examples:
      - key='name', value='Alex'
      - key='answer_style', value='short and concise'
      - key='major', value='Computer Science'
    """
    try:
        clean_key = key.strip()
        clean_val = value.strip()
        if not clean_key or not clean_val:
            return "Failed to remember: Key and value must not be empty."

        set_preference(clean_key, clean_val)
        return f"Saved to memory: {clean_key} = '{clean_val}'."
    except Exception as e:
        return f"Failed to save preference: {str(e)}"


# ==============================================================================
# 6. Tool Schemas (OpenAI Compatible) & Execution Dispatcher
# ==============================================================================
TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather conditions (temperature, humidity, wind) for any city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'Rajkot', 'London', 'New York'",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Safely evaluate mathematical and percentage calculations (e.g. '15% of 2400', 'sqrt(144) + 25 * 4').",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression string to evaluate",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_note",
            "description": "Save a personal note or memo for the user to the database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title or subject of the note (optional)",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full text content of the note",
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_notes",
            "description": "List all previously saved user notes from the database.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_reminder",
            "description": "Add a task reminder with an optional due date/time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The reminder task description, e.g. 'Submit lab report'",
                    },
                    "due_time": {
                        "type": "string",
                        "description": "When the reminder is due, e.g. 'Friday', 'Tomorrow at 5 PM'",
                    },
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_preference",
            "description": "Save a personal user fact or preference into long-term memory (e.g. user's name, communication style, or university).",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Trait or attribute name, e.g. 'name', 'response_style', 'major'",
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to remember, e.g. 'Parth', 'short and concise', 'Computer Science'",
                    },
                },
                "required": ["key", "value"],
            },
        },
    },
]


def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    """
    Central dispatcher that maps tool names to their corresponding Python functions.
    Returns the string result of the tool execution.
    """
    if name == "get_weather":
        return get_weather(arguments.get("city", ""))
    elif name == "calculator":
        return calculator(arguments.get("expression", ""))
    elif name == "add_note":
        return add_note(
            title=arguments.get("title", ""),
            content=arguments.get("content", arguments.get("title", "")),
        )
    elif name == "list_notes":
        return list_notes()
    elif name == "add_reminder":
        return add_reminder(
            task=arguments.get("task", ""),
            due_time=arguments.get("due_time", ""),
        )
    elif name == "remember_preference":
        return remember_preference(
            key=arguments.get("key", ""),
            value=arguments.get("value", ""),
        )
    else:
        return f"Unknown tool: '{name}'"
