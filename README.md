# Nova: AI Virtual Assistant

Nova is a lightweight, explainable AI virtual assistant built for university lab assignments and viva defense.
It runs as a single Streamlit web application and connects to any OpenAI-compatible LLM provider (Groq, Ollama, OpenRouter, OpenAI), allowing models to be swapped by changing a single setting.

---

## 🌟 Key Features

1. **Streaming Chat with SQLite Persistence**:
   - Natural, real-time token streaming directly to the chat interface.
   - Preserves conversation history across app restarts using local SQLite storage.
2. **📄 PDF Document Upload & Q&A**:
   - Upload any lecture notes, research papers, syllabus, or manuals via the sidebar.
   - Extracts document text using `pypdf` with page markers (`--- Page X ---`).
   - Chat with Nova about your uploaded document: request summaries, ask specific questions, or extract concepts with page references.
3. **🛠️ Autonomous Agentic Tool Calling (ReAct Loop)**:
   - **`get_weather(city)`**: Real-time weather and temperature using the free Open-Meteo REST API (no key required).
   - **`calculator(expression)`**: Safe mathematical expression evaluation using Python's AST parser (rejects dangerous functions, no `eval()`).
   - **`add_note(title, content)` & `list_notes()`**: Save and retrieve personal study memos directly from SQLite.
   - **`add_reminder(task, due_time)`**: Track task deadlines and schedules.
   - **`remember_preference(key, value)`**: Remembers your name, interests, and preferred communication style between sessions.
4. **🔊 Spoken Replies (Voice Narration)**:
   - Optional voice narration using Microsoft Edge's free neural Text-to-Speech (`edge-tts`).
   - Cleans formatting (removes markdown symbols, tables, and code blocks) before synthesis.
   - Autoplays smoothly in the background via a hidden HTML5 audio element.
5. **Interactive Sidebar Storage Dashboard**:
   - View, manage, and delete saved user memories, personal notes, and reminders with one-click buttons.
   - Active document badge with page count and a one-click "Remove PDF" button.

---

## 🏗️ Project Architecture

```
├── app.py              # Streamlit user interface, PDF file handler, and chat loop
├── app_config.py       # Configuration loader (.env and Streamlit Secrets)
├── config.py           # Backward-compatible alias for app_config.py
├── database.py         # SQLite connection and CRUD operations (messages, notes, reminders, preferences)
├── llm.py              # OpenAI-compatible ReAct agentic loop with tool calling and PDF context
├── tools.py            # Tool implementations (weather, AST calculator, notes, reminders, preferences)
├── voice.py            # Edge-TTS speech generation, markdown cleaner, and audio handler
├── requirements.txt    # Python dependencies
└── README.md           # Documentation and Viva Defense Guide
```

---

## 🚀 Quickstart & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Create a `.env` file in the project root:
```env
# LLM Provider Configuration (Defaults to Groq's fast Llama 3.3 70B)
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=gsk_your_groq_api_key_here
MODEL_ID=llama-3.3-70b-versatile

# Text-to-Speech Voice
TTS_VOICE=en-US-AriaNeural
```

*(Note: On Streamlit Community Cloud, you can enter these exact keys in the **Secrets** settings without committing a `.env` file).*

### 3. Run the Application
```bash
streamlit run app.py
```

---
