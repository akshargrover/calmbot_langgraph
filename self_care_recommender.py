def suggest_care(state):
    emotions = state["emotions"]
    
    suggestions = {
        "anxiety": "Try a 4-7-8 breathing exercise or take a walk.",
        "joy": "Reflect on what brought you joy. Capture it in a gratitude journal.",
        "gratitude": "Write a short thank-you message to someone today.",
    }
    
    suggestion = suggestions.get(emotions.lower(), "Take a moment to check in with yourself.")
    
    return {**state, "suggestion": suggestion}
