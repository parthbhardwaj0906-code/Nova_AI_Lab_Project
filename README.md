# Nova: AI Virtual Assistant

Nova is a simple, explainable AI assistant built for a university assignment.
It runs as a single Streamlit app and talks to any OpenAI-compatible LLM
provider (Groq, Ollama, OpenRouter and others), so the model can be swapped
by changing one setting.

**Features**
- Streaming chat with saved history
- Tool use: weather, notes, reminders, calculator
- Remembers your name and preferences between sessions
- Optional spoken replies (text-to-speech)

**Tech stack:** Python, Streamlit, SQLite, OpenAI-compatible API, edge-tts

---

## Quickstart

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   Create a `.env` file from the provided `.env.example` and set your LLM credentials:
   ```bash
   cp .env.example .env
   ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```
