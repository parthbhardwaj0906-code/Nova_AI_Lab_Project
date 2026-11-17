"""
health_tools.py - Nova Health Clinical & Wellness Tool Suite

Provides evidence-informed health calculations and tracking tools:
1. calculate_bmi: BMI and WHO classification with screening caveats.
2. estimate_daily_calories: Mifflin-St Jeor equation enforcing safe floors (1200/1500 kcal).
3. log_health_metric: Plausibility validation and persistence for biometric metrics.
4. get_health_history: Formatted history and trend analysis (up/down/stable).
5. add_medication_reminder / list_medication_reminders: User-defined medication scheduler.
6. log_symptom / get_symptom_history: Structured symptom timeline for physician sharing.
7. lookup_nutrition: Open Food Facts search for macronutrients per 100g.
8. remember_health_preference: Long-term profile memory.

All tools return human-readable strings to be observed by the LLM ReAct loop.
"""

from typing import Dict, Any, List
import requests
from database import (
    save_health_metric,
    get_health_metrics,
    save_symptom,
    get_symptoms,
    save_medication_reminder,
    get_medication_reminders,
    delete_medication_reminder as db_delete_medication_reminder,
    set_preference,
)

# Open Food Facts user agent header required by their API policy
OPEN_FOOD_FACTS_HEADERS = {
    "User-Agent": "NovaHealth - Academic Lab Project - Contact: student@univ.edu"
}

# Metric plausibility thresholds to prevent absurd or mistaken inputs
METRIC_LIMITS = {
    "weight": (20.0, 450.0),       # kg
    "sleep_hours": (0.0, 24.0),     # hours
    "water_ml": (0.0, 10000.0),     # ml
    "blood_sugar": (20.0, 600.0),   # mg/dL
    "mood": (1.0, 10.0),            # scale 1-10
}


# ==============================================================================
# 1. Body Mass Index (BMI) Tool
# ==============================================================================
def calculate_bmi(weight_kg: float, height_cm: float) -> str:
    """
    Calculates Body Mass Index (BMI) and provides the standard WHO category.
    Includes clinical caveats noting that BMI is a screening metric, not diagnostic.
    """
    try:
        w = float(weight_kg)
        h = float(height_cm)

        if w <= 0 or h <= 0:
            return "Error: Weight and height must be positive numbers."
        if h < 50 or h > 280:
            return f"Error: Height of {h} cm appears unrealistic. Please enter height in centimeters (e.g. 175)."
        if w < 20 or w > 450:
            return f"Error: Weight of {w} kg appears unrealistic."

        # BMI = weight (kg) / height (m)^2
        height_m = h / 100.0
        bmi = round(w / (height_m ** 2), 1)

        # Standard WHO classification
        if bmi < 18.5:
            category = "Underweight"
        elif 18.5 <= bmi < 25.0:
            category = "Normal weight"
        elif 25.0 <= bmi < 30.0:
            category = "Overweight"
        else:
            category = "Obesity"

        return (
            f"BMI Result: {bmi} kg/m² (Category: {category}).\n"
            f"Clinical Note: BMI is a population-level screening indicator. It does not directly "
            f"measure body fat, muscle mass, or metabolic health. Discuss with your physician for individual context."
        )
    except Exception as err:
        return f"BMI Calculation error: {str(err)}"


# ==============================================================================
# 2. Mifflin-St Jeor Daily Calorie Estimation Tool
# ==============================================================================
def estimate_daily_calories(
    age: int,
    sex: str,
    weight_kg: float,
    height_cm: float,
    activity_level: str,
    goal: str = "maintain",
) -> str:
    """
    Estimates daily caloric requirements using the validated Mifflin-St Jeor formula.
    Strictly enforces safe physiological floors: 1200 kcal/day for women, 1500 kcal/day for men.
    """
    try:
        a = int(age)
        w = float(weight_kg)
        h = float(height_cm)
        s = sex.strip().lower()
        act = activity_level.strip().lower()
        g = goal.strip().lower()

        if a < 12 or a > 120:
            return "Error: Age must be between 12 and 120."
        if w < 25 or h < 80:
            return "Error: Weight or height values are out of plausible range."

        # Mifflin-St Jeor Basal Metabolic Rate (BMR)
        if s.startswith("m"):
            bmr = (10 * w) + (6.25 * h) - (5 * a) + 5
            safe_floor = 1500
        else:
            bmr = (10 * w) + (6.25 * h) - (5 * a) - 161
            safe_floor = 1200

        # Physical activity multiplier
        activity_multipliers = {
            "sedentary": 1.2,          # Desk job, little to no exercise
            "light": 1.375,            # Light exercise 1-3 days/week
            "moderate": 1.55,          # Moderate exercise 3-5 days/week
            "very_active": 1.725,      # Hard exercise 6-7 days/week
            "extra_active": 1.9,       # Very hard daily exercise/physical job
        }
        mult = activity_multipliers.get(act, 1.375)
        tdee = bmr * mult  # Total Daily Energy Expenditure

        # Goal adjustments
        if "loss" in g or "deficit" in g:
            target = tdee - 400
            goal_label = "Gentle weight loss (deficit)"
        elif "gain" in g or "surplus" in g:
            target = tdee + 400
            goal_label = "Muscle gain / weight gain (surplus)"
        else:
            target = tdee
            goal_label = "Weight maintenance"

        target_calories = int(round(target))
        floor_applied = False

        # Enforce safe minimum calorie floors
        if target_calories < safe_floor:
            target_calories = safe_floor
            floor_applied = True

        response = (
            f"Daily Calorie Estimate (Mifflin-St Jeor Formula):\n"
            f"- Basal Metabolic Rate (BMR): {int(round(bmr))} kcal/day\n"
            f"- Maintenance Energy (TDEE): {int(round(tdee))} kcal/day\n"
            f"- Recommended Target ({goal_label}): {target_calories} kcal/day\n"
        )

        if floor_applied:
            response += (
                f"\n⚠️ Safety Guardrail Applied: Target adjusted up to the clinically safe minimum "
                f"floor of {safe_floor} kcal/day ({'men' if s.startswith('m') else 'women'}) to prevent "
                f"nutritional deficiency, muscle wasting, and metabolic slowdown."
            )

        return response
    except Exception as err:
        return f"Calorie Estimation error: {str(err)}"


# ==============================================================================
# 3. Health Metric Logging & History
# ==============================================================================
def log_health_metric(
    metric: str,
    value: float,
    unit: str,
    user_id: str = "default_user",
) -> str:
    """
    Validates and stores a health metric (weight, sleep, water, blood pressure, etc.).
    """
    try:
        clean_metric = metric.strip().lower()
        val = float(value)

        # Validate physiological plausibility
        if clean_metric in METRIC_LIMITS:
            min_val, max_val = METRIC_LIMITS[clean_metric]
            if val < min_val or val > max_val:
                return (
                    f"Warning: A {clean_metric} of {val} {unit} is outside plausible physiological "
                    f"limits ({min_val} - {max_val} {unit}). Please verify your input."
                )

        metric_id = save_health_metric(user_id, clean_metric, val, unit)
        return f"Logged {clean_metric}: {val} {unit} (Entry #{metric_id} saved for {user_id})."
    except Exception as err:
        return f"Failed to log metric: {str(err)}"


def get_health_history(
    metric: str,
    days: int = 14,
    user_id: str = "default_user",
) -> str:
    """
    Retrieves recent entries for a metric and computes a trend direction.
    """
    try:
        clean_metric = metric.strip().lower()
        records = get_health_metrics(user_id, metric=clean_metric, limit=20)

        if not records:
            return f"No logged entries found for '{clean_metric}' under profile '{user_id}'."

        lines = [f"- {r['created_at'][:10]}: {r['value']} {r['unit']}" for r in records[:7]]

        # Compute trend direction if at least 2 entries exist
        trend_note = "Trend: Insufficient data for trend."
        if len(records) >= 2:
            latest = records[0]["value"]
            previous = records[1]["value"]
            diff = latest - previous
            if abs(diff) < 0.05 * previous:
                trend_note = "Trend: Stable over recent logs."
            elif diff > 0:
                trend_note = f"Trend: Upward (+{round(diff, 2)} {records[0]['unit']})."
            else:
                trend_note = f"Trend: Downward ({round(diff, 2)} {records[0]['unit']})."

        return (
            f"Health History for '{clean_metric}':\n"
            + "\n".join(lines)
            + f"\n\n{trend_note}"
        )
    except Exception as err:
        return f"Failed to retrieve health history: {str(err)}"


# ==============================================================================
# 4. Medication Reminders (No Dosage Suggestions)
# ==============================================================================
def add_medication_reminder(
    name: str,
    time: str,
    notes: str = "",
    user_id: str = "default_user",
) -> str:
    """
    Saves a user-specified medication schedule time.
    Nova Health strictly records the user's schedule and never recommends dosages.
    """
    try:
        clean_name = name.strip()
        clean_time = time.strip()
        if not clean_name:
            return "Error: Medication name cannot be empty."

        rem_id = save_medication_reminder(user_id, clean_name, clean_time, notes)
        return (
            f"Medication reminder saved: '{clean_name}' at '{clean_time}'. "
            f"(Reminder #{rem_id} created for {user_id})."
        )
    except Exception as err:
        return f"Failed to add medication reminder: {str(err)}"


def list_medication_reminders(user_id: str = "default_user") -> str:
    """Retrieves all active medication reminders for the user."""
    try:
        reminders = get_medication_reminders(user_id)
        if not reminders:
            return f"No medication reminders currently scheduled for profile '{user_id}'."

        lines = []
        for r in reminders:
            note_str = f" (Notes: {r['notes']})" if r["notes"] else ""
            lines.append(f"- [#{r['id']}] {r['name']} at {r['time']}{note_str}")

        return "Active Medication Schedule:\n" + "\n".join(lines)
    except Exception as err:
        return f"Failed to list medication reminders: {str(err)}"


def delete_medication_reminder(reminder_id: int) -> str:
    """Deletes a medication reminder by its ID."""
    try:
        db_delete_medication_reminder(reminder_id)
        return f"Medication reminder #{reminder_id} successfully deleted."
    except Exception as err:
        return f"Failed to delete reminder: {str(err)}"


# ==============================================================================
# 5. Symptom Logging & Clinical Timeline
# ==============================================================================
def log_symptom(
    symptom: str,
    severity_1_to_10: int,
    notes: str = "",
    user_id: str = "default_user",
) -> str:
    """
    Logs an experienced symptom with a severity score (1=very mild, 10=extreme).
    """
    try:
        clean_symptom = symptom.strip()
        sev = int(severity_1_to_10)

        if not clean_symptom:
            return "Error: Symptom description cannot be empty."
        if sev < 1 or sev > 10:
            return "Error: Severity score must be between 1 (mild) and 10 (severe)."

        symptom_id = save_symptom(user_id, clean_symptom, sev, notes)
        return (
            f"Logged symptom: '{clean_symptom}' (Severity: {sev}/10) saved under #{symptom_id}.\n"
            f"Reminder: If this symptom is severe, unexpected, or worsening, please consult a medical professional."
        )
    except Exception as err:
        return f"Failed to log symptom: {str(err)}"


def get_symptom_history(days: int = 30, user_id: str = "default_user") -> str:
    """
    Returns a structured symptom timeline formatted for the user to review with a doctor.
    """
    try:
        entries = get_symptoms(user_id, limit=days)
        if not entries:
            return f"No logged symptoms found for profile '{user_id}'."

        lines = []
        for e in entries:
            date_str = e["created_at"][:16]
            note_str = f" | Notes: {e['notes']}" if e["notes"] else ""
            lines.append(f"- [{date_str}] {e['symptom']} (Severity: {e['severity']}/10){note_str}")

        return (
            f"Clinical Symptom Log (for profile '{user_id}'):\n"
            + "\n".join(lines)
            + "\n\n💡 You can bring this summary to your next healthcare appointment."
        )
    except Exception as err:
        return f"Failed to retrieve symptom history: {str(err)}"


# ==============================================================================
# 6. Open Food Facts Nutrition Lookup Tool
# ==============================================================================
def lookup_nutrition(food: str) -> str:
    """
    Queries the Open Food Facts free database for calories and macronutrients per 100g.
    Requires no API key.
    """
    clean_food = food.strip()
    if not clean_food:
        return "Error: Please specify a food item to look up."

    # Try both endpoints in case of regional CDN downtime
    endpoints = [
        f"https://world.openfoodfacts.net/cgi/search.pl?search_terms={clean_food}&search_simple=1&action=process&json=1",
        f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={clean_food}&search_simple=1&action=process&json=1",
    ]

    for url in endpoints:
        try:
            res = requests.get(url, headers=OPEN_FOOD_FACTS_HEADERS, timeout=6)
            if res.status_code == 200:
                data = res.json()
                products = data.get("products", [])
                if products:
                    for p in products[:3]:
                        n = p.get("nutriments", {})
                        kcal = n.get("energy-kcal_100g") or n.get("energy-kcal")
                        if kcal is not None:
                            product_name = p.get("product_name") or clean_food
                            protein = n.get("proteins_100g", "N/A")
                            carbs = n.get("carbohydrates_100g", "N/A")
                            fat = n.get("fat_100g", "N/A")
                            serving = p.get("serving_size", "100g")

                            return (
                                f"Nutrition Information for '{product_name}' (per 100g):\n"
                                f"- Calories: {round(float(kcal), 1)} kcal\n"
                                f"- Protein: {protein} g\n"
                                f"- Carbohydrates: {carbs} g\n"
                                f"- Fat: {fat} g\n"
                                f"- Serving reference: {serving}\n"
                                f"(Data source: Open Food Facts)"
                            )
        except Exception:
            continue

    return f"Could not find nutritional details for '{clean_food}' in the open database."


# ==============================================================================
# 7. User Health Profile & Preference Memory
# ==============================================================================
def remember_health_preference(key: str, value: str, user_id: str = "default_user") -> str:
    """
    Saves a user health preference or trait (e.g. dietary restrictions, age, sex, fitness goal).
    """
    try:
        clean_key = key.strip()
        clean_val = value.strip()
        if not clean_key or not clean_val:
            return "Error: Preference key and value must not be empty."

        set_preference(user_id, clean_key, clean_val)
        return f"Saved to health profile ({user_id}): {clean_key} = '{clean_val}'."
    except Exception as err:
        return f"Failed to save profile preference: {str(err)}"


# ==============================================================================
# OpenAI Function Calling Tools Schema
# ==============================================================================
HEALTH_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculate_bmi",
            "description": "Calculate Body Mass Index (BMI) and WHO category from weight (kg) and height (cm).",
            "parameters": {
                "type": "object",
                "properties": {
                    "weight_kg": {"type": "number", "description": "Weight in kilograms (e.g. 70.5)"},
                    "height_cm": {"type": "number", "description": "Height in centimeters (e.g. 175)"},
                },
                "required": ["weight_kg", "height_cm"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_daily_calories",
            "description": "Estimate daily calorie maintenance and target using Mifflin-St Jeor equation. Enforces safe minimum floors (1200 kcal women, 1500 kcal men).",
            "parameters": {
                "type": "object",
                "properties": {
                    "age": {"type": "integer", "description": "Age in years"},
                    "sex": {"type": "string", "enum": ["male", "female"], "description": "Biological sex for BMR calculation"},
                    "weight_kg": {"type": "number", "description": "Weight in kilograms"},
                    "height_cm": {"type": "number", "description": "Height in centimeters"},
                    "activity_level": {
                        "type": "string",
                        "enum": ["sedentary", "light", "moderate", "very_active"],
                        "description": "Physical activity frequency",
                    },
                    "goal": {
                        "type": "string",
                        "enum": ["maintain", "gentle_loss", "muscle_gain"],
                        "description": "Calorie goal",
                    },
                },
                "required": ["age", "sex", "weight_kg", "height_cm", "activity_level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_health_metric",
            "description": "Log a physiological metric (weight, sleep_hours, water_ml, blood_sugar, blood_pressure, mood).",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["weight", "sleep_hours", "water_ml", "blood_sugar", "blood_pressure", "mood"],
                        "description": "Type of metric to log",
                    },
                    "value": {"type": "number", "description": "Numeric measurement value"},
                    "unit": {"type": "string", "description": "Measurement unit (e.g. 'kg', 'hours', 'ml', 'mg/dL')"},
                },
                "required": ["metric", "value", "unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_health_history",
            "description": "Retrieve recent history and trend direction for a health metric.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "description": "Metric name, e.g. 'weight' or 'sleep_hours'"},
                    "days": {"type": "integer", "description": "Number of days/entries to inspect (default 14)"},
                },
                "required": ["metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_medication_reminder",
            "description": "Schedule a user-specified medication reminder time. Never calculates or advises dosages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Medication or supplement name"},
                    "time": {"type": "string", "description": "Scheduled time, e.g. '8:00 AM' or 'with dinner'"},
                    "notes": {"type": "string", "description": "Optional notes from user (e.g. 'take with water')"},
                },
                "required": ["name", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_medication_reminders",
            "description": "View all current medication reminders.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_symptom",
            "description": "Record an experienced symptom with a 1-10 severity rating and notes for physician review.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symptom": {"type": "string", "description": "Symptom description (e.g. 'mild headache', 'knee soreness')"},
                    "severity_1_to_10": {"type": "integer", "description": "Severity rating from 1 (mild) to 10 (severe)"},
                    "notes": {"type": "string", "description": "Additional context (e.g. 'after running', 'lasted 2 hours')"},
                },
                "required": ["symptom", "severity_1_to_10"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_symptom_history",
            "description": "Get a clean summary timeline of recent symptoms formatted for sharing with a doctor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Number of days/entries to review (default 30)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_nutrition",
            "description": "Look up calories, protein, carbs, and fat per 100g for a food item using Open Food Facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "food": {"type": "string", "description": "Food name (e.g. 'oatmeal', 'banana', 'greek yogurt')"},
                },
                "required": ["food"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_health_preference",
            "description": "Save personal user health profile information (e.g. age, dietary preferences, fitness goals).",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Profile attribute, e.g. 'dietary_preference', 'primary_goal', 'age'"},
                    "value": {"type": "string", "description": "Attribute value, e.g. 'vegetarian', 'improve sleep', '24'"},
                },
                "required": ["key", "value"],
            },
        },
    },
]


def execute_health_tool(name: str, arguments: Dict[str, Any], user_id: str = "default_user") -> str:
    """
    Central dispatcher that invokes the corresponding Python health function.
    """
    if name == "calculate_bmi":
        return calculate_bmi(
            weight_kg=arguments.get("weight_kg", 0),
            height_cm=arguments.get("height_cm", 0),
        )
    elif name == "estimate_daily_calories":
        return estimate_daily_calories(
            age=arguments.get("age", 25),
            sex=arguments.get("sex", "female"),
            weight_kg=arguments.get("weight_kg", 65),
            height_cm=arguments.get("height_cm", 165),
            activity_level=arguments.get("activity_level", "moderate"),
            goal=arguments.get("goal", "maintain"),
        )
    elif name == "log_health_metric":
        return log_health_metric(
            metric=arguments.get("metric", ""),
            value=arguments.get("value", 0),
            unit=arguments.get("unit", ""),
            user_id=user_id,
        )
    elif name == "get_health_history":
        return get_health_history(
            metric=arguments.get("metric", ""),
            days=arguments.get("days", 14),
            user_id=user_id,
        )
    elif name == "add_medication_reminder":
        return add_medication_reminder(
            name=arguments.get("name", ""),
            time=arguments.get("time", ""),
            notes=arguments.get("notes", ""),
            user_id=user_id,
        )
    elif name == "list_medication_reminders":
        return list_medication_reminders(user_id=user_id)
    elif name == "log_symptom":
        return log_symptom(
            symptom=arguments.get("symptom", ""),
            severity_1_to_10=arguments.get("severity_1_to_10", 5),
            notes=arguments.get("notes", ""),
            user_id=user_id,
        )
    elif name == "get_symptom_history":
        return get_symptom_history(
            days=arguments.get("days", 30),
            user_id=user_id,
        )
    elif name == "lookup_nutrition":
        return lookup_nutrition(food=arguments.get("food", ""))
    elif name == "remember_health_preference":
        return remember_health_preference(
            key=arguments.get("key", ""),
            value=arguments.get("value", ""),
            user_id=user_id,
        )
    else:
        return f"Unknown tool: '{name}'"
