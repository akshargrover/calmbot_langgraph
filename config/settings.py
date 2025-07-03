import os
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')

# Load Gemini API key from environment or config.yaml

def get_gemini_api_key():
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        return api_key
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('GEMINI_API_KEY')
    raise ValueError('Gemini API key not found. Set GEMINI_API_KEY env variable or config.yaml.')

# Example: Other constants
EMBEDDING_DIM = 384  # Example, adjust as needed
FAISS_INDEX_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'faiss_index')
USER_LOGS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'user_logs')
