# AGENTS.md — LLM-Chatbot

## Project Purpose
This project is a production-grade LLM-powered chatbot built with Streamlit and the Google Gemini API.
It supports multi-turn conversations with buffer memory, long-term memory stored in PostgreSQL with pgvector,
and Model Armor guardrails for safe, policy-compliant responses.

---

## Tech Stack
- **Frontend:** Streamlit
- **LLM:** Google Gemini API (gemini-1.5-pro or gemini-2.0-flash)
- **Short-term Memory:** Conversation buffer (last N messages)
- **Long-term Memory:** PostgreSQL + pgvector (semantic similarity search)
- **Safety:** Google Model Armor guardrails
- **Deployment:** Google Cloud Run

---

## Build Instructions

### Prerequisites
- Python 3.11+
- PostgreSQL instance with pgvector extension enabled
- Google Cloud project with Gemini API enabled
- Model Armor API enabled

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file:
```
GEMINI_API_KEY=your_gemini_api_key
DATABASE_URL=postgresql://user:password@host:5432/chatbot_db
MODEL_ARMOR_ENDPOINT=your_model_armor_endpoint
```

### Run Locally
```bash
streamlit run app.py
```

### Run with Docker
```bash
docker build -t llm-chatbot .
docker run -p 8501:8501 llm-chatbot
```

---

## Coding Standards
- Follow PEP 8 style guide
- All functions must have docstrings
- Use type hints on all function signatures
- Keep functions under 50 lines; refactor if longer
- Use environment variables for all secrets (never hardcode)
- Use `logging` module instead of `print` statements
- All database interactions must use parameterized queries

---

## Project Structure
```
LLM-Chatbot/
├── app.py                  # Main Streamlit app entry point
├── memory/
│   ├── buffer_memory.py    # Short-term conversation buffer
│   └── long_term_memory.py # PostgreSQL + pgvector memory
├── guardrails/
│   └── model_armor.py      # Model Armor integration
├── utils/
│   └── gemini_client.py    # Gemini API wrapper
├── tests/
│   ├── test_memory.py
│   ├── test_guardrails.py
│   └── test_gemini_client.py
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

---

## Testing Commands
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=term-missing

# Run a specific test file
pytest tests/test_memory.py -v
```

---

## Deployment Expectations
- Deploy to **Google Cloud Run**
- Container must listen on port **8501**
- All secrets injected via **Cloud Run environment variables** or **Secret Manager**
- Health check endpoint: `/healthz`
- The app must start within **60 seconds**
- Memory usage must stay under **512MB**

---

## Key Features to Implement
1. **Buffer Memory** — Store last 10 conversation turns in session state
2. **Long-term Memory** — Embed user messages with Gemini embeddings; store in pgvector; retrieve top-3 similar memories on each new message
3. **Model Armor** — Pass all user inputs through Model Armor before sending to Gemini; block unsafe content
4. **Token Tracking** — Use `GenerativeModel.count_tokens()` to track input/output tokens per message; display in sidebar
