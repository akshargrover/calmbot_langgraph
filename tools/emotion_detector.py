from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import get_gemini_api_key
import re

def detect_emotion(state):
    user_text = state["text"]
    if isinstance(user_text, list):
        user_text = " ".join(x.content if hasattr(x, "content") else str(x) for x in user_text)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=get_gemini_api_key())
    prompt = (
        f"""
    Analyze the following text and identify the user's primary emotion.\n"
    Choose from: anxiety, joy, shame, gratitude, sadness, anger, fear, surprise, or other.\n"
    Respond ONLY with a single line of valid JSON in this format: {{\"emotion\": <emotion>, \"confidence\": <0-1>, \"details\": <short reasoning>}}\n"
    Do not include any explanation or extra text.\n"
    Text: {user_text}
    """
    )
    response = llm.invoke(prompt)
    import json
    result = None
    try:
        result = json.loads(response.content)
        print("LLM response (parsed as JSON): ", response.content)
    except json.JSONDecodeError:
        # Try to extract JSON object from the response using regex
        print("LLM response (raw): ", response.content)
        match = re.search(r'\{.*?\}', response.content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
            except Exception:
                result = None
        if not result:
            result = {"emotion": "other", "confidence": 0.5, "details": "Could not parse emotion"}
    prev_emotion = state.get("emotions", None)
    prev_confidence = state.get("confidence", None)
    # Only update if emotion is not 'other' and confidence is high
    if result["emotion"] != "other" and result["confidence"] > 0.6:
        state.update({
            "emotions": result["emotion"],
            "confidence": result["confidence"],
            "details": result.get("details", "")
        })
    else:
        # Retain previous emotion if available
        state.update({
            "emotions": prev_emotion,
            "confidence": prev_confidence,
            "details": result.get("details", "")
        })
    return state
