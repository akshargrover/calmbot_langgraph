from vertexai.language_models import ChatModel

def detect_emotion(state):
    user_input = state["input"]
    
    model = ChatModel.from_pretrained("gemini-1.5-flash")  # Gemini 2.0 Flash
    response = model.predict(f"Classify this text into nuanced emotions: {user_input}")
    
    return {"emotions": response.text, "input": user_input}
