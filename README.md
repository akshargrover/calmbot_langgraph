# Calmbot (Built using LangGraph)

A conversational mental health assistant powered by LangGraph, LLMs, and Retrieval-Augmented Generation (RAG). The bot detects user emotions, provides self-care suggestions, matches users to therapists, and offers crisis support, all orchestrated through a modular graph-based workflow.

## Features

- **Emotion Detection:** Analyzes user input to identify primary emotions (anxiety, joy, shame, gratitude, sadness, anger, fear, surprise, or other).
- **Self-Care Web Search:** Finds real-time self-care strategies for the detected emotion.
- **Self-Care RAG:** Retrieves personalized self-care suggestions from an internal RAG (vector) database.
- **Therapist Matching (RAG):** Matches users to the best therapist using a RAG-based approach.
- **Self-Care Recommendation:** Offers actionable self-care tips based on the detected emotion.
- **Appointment Booking:** Offers and books appointments with therapists based on user needs and emotion.
- **Crisis Detection & Response:** Detects crisis language and provides immediate support and helpline information.
- **User Memory:** Retrieves similar past moods and user history to personalize responses.
- **Graph-based Workflow:** Modular, extensible workflow using LangGraph.
- **Modern UI:** Streamlit-based chat interface for easy interaction.

## Directory Structure

```
.
├── app_streamlit.py           # Streamlit frontend app
├── main.py                    # FastAPI backend entry point
├── graph_builder.py           # LangGraph workflow definition
├── rag_index.py               # RAG index utilities
├── requirements.txt           # Python dependencies
├── README.md
├── LICENSE
├── config/
│   ├── config.yaml            # Model and API config
│   └── settings.py            # API key and settings loader
├── data/
│   ├── therapist_profiles.json
│   ├── therapist.db           # Therapist/availability DB
│   ├── therapist_rag/         # RAG vector index for therapists
│   │   ├── index.faiss
│   │   └── index.pkl
│   ├── selfcare_pdfs/         # Source PDFs for self-care
│   │   ├── Life-in-Mind-Self-care.pdf
│   │   ├── SelfCareReportR13.pdf
│   │   ├── A Comprehensive Guide to Coping Strategies.pdf
│   │   ├── A Guide to Understanding and Managing Anxiety.pdf
│   │   └── A Guide to Understanding and Managing Depression.pdf
│   ├── selfcare_rag/          # RAG vector index for self-care
│   │   ├── index.faiss
│   │   └── index.pkl
│   ├── user_logs/             # User logs
│   │   ├── demo_user.jsonl
│   │   └── ...
│   └── faiss_index/           # User memory vector index (optional/empty)
├── tools/
│   ├── agent_router.py        # Orchestrates tool selection
│   ├── appointment_tool.py    # Appointment/booking logic
│   ├── crisis_responder.py    # Crisis support
│   ├── emotion_detector.py    # Emotion detection
│   ├── memory_store.py        # User memory/history
│   ├── self_care_websearch.py # Self-care web search
│   ├── selfcare_rag_suggester.py # RAG self-care
│   └── __init__.py
├── utils/
│   ├── build_selfcare_rag_index.py
│   ├── build_therapist_rag_index.py
│   ├── config_loader.py
│   ├── db_seed.py
│   ├── embedding.py
│   ├── faiss_utils.py
│   ├── model_loader.py
│   └── data_schema.py
├── notebooks/                 # (Optional) Jupyter notebooks
└── ...
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd calmbot_langgraph
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables:**
   - Create a `.env` file in the root directory.
   - Add your Gemini API key:
     ```
     GEMINI_API_KEY=your_gemini_api_key_here
     ```
   - (Optional) Add OpenAI or Groq API keys if using those models.
4. **Seed the database (optional, for demo data):**
   ```bash
   python utils/db_seed.py
   ```
5. **Build RAG indexes (optional, for best results):**
   ```bash
   python utils/build_selfcare_rag_index.py
   python utils/build_therapist_rag_index.py
   ```

## Running the App

Start the FastAPI backend:
```bash
uvicorn main:app --reload
```

Start the Streamlit frontend (in a separate terminal):
```bash
streamlit run app_streamlit.py
```

- Visit [http://localhost:8501](http://localhost:8501) in your browser for the chat UI.
- The Streamlit app communicates with the FastAPI backend at [http://localhost:8000](http://localhost:8000).
- Use the `/analyze` endpoint (POST) to interact programmatically.

## API Usage Example

```bash
curl -X POST -H "Content-Type: application/json" -d '{"user_input": "I feel anxious and overwhelmed"}' http://localhost:8000/analyze
```

Response JSON includes:
- `agent_message`: Main response from the bot
- `needs_clarification`: Whether the bot needs more info
- `waiting_for_input`: If the bot is waiting for a specific input
- `expected_input`: What input is expected next
- `appointment_stage`: Current appointment booking stage
- `emotion`: Detected emotion
- `forecast`: Mood forecast
- `rag_self_care`: Personalized self-care
- `prompt`: Supportive prompt
- `care_suggestion`: Actionable tip
- `appointment_offer`: Therapist booking offer
- `appointment_status`: Booking result
- `therapist_match`: RAG therapist match
- `agent_router_output`: Router output
- `router_trace`: Router trace/debug info
- `crisis_response`: Crisis support message
- `next_action`: Next action for the user
- `debug_info`: Debugging details

## Main Functions & Tools

- **Emotion Detection:** Classifies user emotion using LLMs.
- **Self-Care Web Search:** Fetches real-time self-care strategies.
- **Self-Care RAG:** Retrieves self-care from internal vector DB.
- **Therapist Match (RAG):** Suggests best therapist using RAG.
- **Self-Care Recommendation:** Offers actionable self-care tips.
- **Appointment Booking:** Books therapist appointments.
- **Crisis Detection & Response:** Detects crisis and provides helplines.
- **User Memory:** Retrieves similar past moods for personalization.
- **Graph-based Workflow:** Modular, extensible workflow using LangGraph.
- **Modern UI:** Streamlit-based chat interface for easy interaction.

## License

MIT License

