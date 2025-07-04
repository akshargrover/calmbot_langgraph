# calmbot_langgraph

A conversational mental health assistant powered by LangGraph, LLMs, and Retrieval-Augmented Generation (RAG). The bot detects user emotions, provides self-care suggestions, matches users to therapists, and offers crisis support, all orchestrated through a modular graph-based workflow.

## Features

- **Emotion Detection:** Analyzes user input to identify primary emotions (anxiety, joy, shame, gratitude, sadness, anger, fear, surprise, or other).
- **Emotion Context Search:** Fetches recent news and stories related to the user's emotion and mental health.
- **Self-Care Web Search:** Finds real-time self-care strategies for the detected emotion.
- **Self-Care RAG:** Retrieves personalized self-care suggestions from an internal RAG (vector) database.
- **Therapist Matching (RAG):** Matches users to the best therapist using a RAG-based approach.
- **Prompt Tailoring:** Crafts supportive, emotion-specific prompts for the user.
- **Self-Care Recommendation:** Offers actionable self-care tips based on the detected emotion.
- **Appointment Booking:** Offers and books appointments with therapists based on user needs and emotion.
- **Crisis Detection & Response:** Detects crisis language and provides immediate support and helpline information.
- **User Memory:** Retrieves similar past moods and user history to personalize responses.

## Directory Structure

```
.
├── app.py                  # Flask web server entry point
├── graph_builder.py        # LangGraph workflow definition
├── rag_index.py            # RAG index utilities
├── requirements.txt        # Python dependencies
├── therapists.db           # SQLite DB for therapist info
├── README.md
├── LICENSE
├── config/
│   ├── config.yaml         # Model and API config
│   └── settings.py         # API key and settings loader
├── data/
│   ├── therapist_profiles.json
│   ├── therapist.db        # Therapist/availability DB
│   ├── therapist_rag/      # RAG vector index for therapists
│   ├── selfcare_pdfs/      # Source PDFs for self-care
│   ├── selfcare_rag/       # RAG vector index for self-care
│   ├── user_logs/          # User logs
│   └── faiss_index/        # User memory vector index
├── tools/
│   ├── agent_router.py         # Orchestrates tool selection
│   ├── appointment_tool.py     # Appointment/booking logic
│   ├── crisis_responder.py     # Crisis support
│   ├── emotion_context_search.py # News/stories search
│   ├── emotion_detector.py     # Emotion detection
│   ├── memory_store.py         # User memory/history
│   ├── prompt_tailor.py        # Supportive prompt generator
│   ├── self_care_recommender.py# Self-care tips
│   ├── self_care_websearch.py  # Self-care web search
│   ├── selfcare_rag_suggester.py # RAG self-care
│   ├── therapist_match_rag.py  # RAG therapist match
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
├── templates/
│   └── index.html              # Web UI
└── notebooks/                  # (Optional) Jupyter notebooks
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

Start the Flask server:
```bash
python app.py
```

- Visit [http://localhost:5000](http://localhost:5000) in your browser for the web UI.
- Use the `/analyze` endpoint (POST) to interact programmatically.

## API Usage Example

```bash
curl -X POST -F "user_input=I feel anxious and overwhelmed" http://localhost:5000/analyze
```

Response JSON includes:
- `emotion`: Detected emotion
- `forecast`: Mood forecast
- `emotion_context_links`: News/stories
- `self_care_articles`: Self-care tips
- `rag_self_care`: Personalized self-care
- `prompt`: Supportive prompt
- `care_suggestion`: Actionable tip
- `appointment_offer`: Therapist booking offer
- `appointment_status`: Booking result
- `therapist_match`: RAG therapist match
- `crisis_response`: Crisis support message

## Main Functions & Tools

- **Emotion Detection:** Classifies user emotion using LLMs.
- **Emotion Context Search:** Finds recent news/stories for the emotion.
- **Self-Care Web Search:** Fetches real-time self-care strategies.
- **Self-Care RAG:** Retrieves self-care from internal vector DB.
- **Therapist Match (RAG):** Suggests best therapist using RAG.
- **Prompt Tailoring:** Crafts supportive, emotion-specific prompts.
- **Self-Care Recommendation:** Offers actionable self-care tips.
- **Appointment Booking:** Books therapist appointments.
- **Crisis Detection & Response:** Detects crisis and provides helplines.
- **User Memory:** Retrieves similar past moods for personalization.

## License

MIT License

