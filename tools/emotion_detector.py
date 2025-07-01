from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import get_gemini_api_key

def detect_emotion(state):
    user_text = state["text"]
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=get_gemini_api_key())
    prompt = (
        """
        Analyze the following text and identify the user's primary emotion. 
        Choose from: anxiety, joy, shame, gratitude, sadness, anger, fear, surprise, or other. 
        Respond in JSON: {"emotion": <emotion>, "confidence": <0-1>, "details": <short reasoning>}.
        Text: """ + user_text
    )
    response = llm.invoke(prompt)
    # Parse response (assume response.content is JSON)
    import json
    result = json.loads(response.content)
    state.update({
        "emotions": result["emotion"],
        "confidence": result["confidence"],
        "details": result.get("details", "")
    })
    return state
