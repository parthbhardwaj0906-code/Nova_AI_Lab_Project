# Nova Health: AI Health & Wellness Assistant

Nova Health is a lightweight, safety-first health and wellness virtual assistant built with Python 3.11, Streamlit, and SQLite. Designed for university demonstrations and viva defense, Nova Health features pre-LLM deterministic emergency screening, evidence-informed wellness calculations, user-partitioned biometric logging, Open Food Facts nutrition lookup, and optional voice narration.

---

## 🛡️ Clinical Safety Architecture (Top Priority)

In clinical AI systems, safety-critical decisions must **never** rely exclusively on non-deterministic LLM generation. Nova Health implements defense-in-depth safety:

1. **Pre-LLM Deterministic Emergency Screener (`safety.py`)**:
   - Executes regex and keyword matching on incoming messages **before** calling the LLM.
   - **Acute Emergencies**: Intercepts chest pain, respiratory distress, stroke signs (FAST: face drooping, slurred speech), severe bleeding, and drug overdoses. Instantly outputs a standardized non-LLM alert directing the user to contact local emergency services immediately using `EMERGENCY_NUMBER` (e.g. 911 / 112).
   - **Crisis & Self-Harm**: Detects suicidal ideation or self-injury and immediately provides compassionate support with direct links to the **988 Suicide & Crisis Lifeline** and crisis text lines.
   - **Disordered Eating Guardrail**: Screens for purging, starvation, or fear of eating, advising professional support rather than diet numbers.
2. **Strict Clinical Boundaries**:
   - **No Diagnoses**: Never claims a user has a specific medical condition.
   - **No Prescription Advice**: Never recommends prescription pharmaceuticals, calculates dosages, or advises altering medications.
   - **Mandatory Escalation**: Urges consulting licensed medical professionals for severe, persistent, or worsening symptoms.
3. **Calorie Deficit Safety Floors**:
   - The calorie estimation tool enforces physiological minimums (**1200 kcal/day for women**, **1500 kcal/day for men**) to protect against metabolic damage and malnutrition.
4. **Permanent Disclaimers & Privacy**:
   - Permanent UI medical disclaimer header.
   - User profile partitioning (`user_id`) to prevent health data contamination across visitors.

---

## 🛠️ Health Tool Suite (`health_tools.py`)

- **`calculate_bmi(weight_kg, height_cm)`**: Calculates Body Mass Index and WHO classification, noting clinical limitations of BMI as a screening tool.
- **`estimate_daily_calories(age, sex, weight_kg, height_cm, activity_level, goal)`**: Implements the validated **Mifflin-St Jeor equation**, strictly enforcing the 1200 / 1500 kcal daily floor.
- **`log_health_metric(metric, value, unit)`**: Validates physiological ranges and logs weight, sleep hours, water intake, blood sugar, and mood.
- **`get_health_history(metric, days)`**: Analyzes recent logs and determines trend direction (upward, downward, stable).
- **`add_medication_reminder(name, time, notes)`**: Schedules user-defined reminders without calculating or suggesting dosages.
- **`log_symptom(symptom, severity_1_to_10, notes)` & `get_symptom_history()`**: Formats a clean clinical timeline designed for users to take to doctor visits.
- **`lookup_nutrition(food)`**: Queries the free [Open Food Facts](https://world.openfoodfacts.org/) search API for calories, protein, carbohydrates, and fat per 100g.

---

## 🚀 Quickstart & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Copy `.env.example` to `.env` and set your preferred provider credentials:
```bash
cp .env.example .env
```
Example configuration for **Groq** (Fast & Free):
```env
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=gsk_yourGroqKeyHere
MODEL_ID=llama-3.3-70b-versatile
EMERGENCY_NUMBER=911
TTS_VOICE=en-US-AriaNeural
```

### 3. Run Nova Health
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔍 File Layout & Viva Breakdown

- **`app.py`**: Streamlit presentation layer. Renders the medical disclaimer, user profile switcher, interactive sidebar dashboards (metric line charts, medication schedule, symptom timeline), and hidden audio player.
- **`safety.py`**: Pre-LLM deterministic emergency and crisis firewall.
- **`health_tools.py`**: Health calculations (BMI, Mifflin-St Jeor), range validation, Open Food Facts API client, and tool dispatching.
- **`database.py`**: SQLite database managing `messages`, `health_metrics`, `symptoms`, `medication_reminders`, and `user_preferences` with `user_id` multi-tenant isolation.
- **`llm.py`**: OpenAI-compatible client orchestrating the ReAct loop with health tool schemas and clinical safety prompt boundaries.
- **`voice.py`**: Text cleaner and Edge-TTS voice synthesis engine.
- **`config.py`**: Environment variables, emergency contact numbers, and core persona definition.

---

## ⚠️ Limitations & Future Work

### Limitations
1. **Local SQLite Storage**: Suitable for personal and lab demonstrations; production healthcare systems require HIPAA/GDPR-compliant encrypted cloud databases.
2. **Single-Agent Architecture**: Uses an iterative ReAct loop; complex multi-symptom triage could benefit from a multi-agent specialist panel.

### Future Work
1. **Wearable Device Integration**: Sync with Apple HealthKit or Google Health Connect for automatic heart rate and step logging.
2. **Document RAG (Medical Knowledge Base)**: Ingest medical reference guides into ChromaDB with BM25 + Reciprocal Rank Fusion (RRF) for evidence-backed lifestyle articles.
3. **Voice Input (Whisper STT)**: Hands-free voice input for accessibility.
