def suggest_care(state):
    emotions = state.get("emotions", "other").lower()
    suggestions = {
        "anxiety": "Try a 4-7-8 breathing exercise or take a walk.",
        "joy": "Reflect on what brought you joy. Capture it in a gratitude journal.",
        "gratitude": "Write a short thank-you message to someone today.",
        "shame": "Practice self-compassion. Remember, mistakes are part of growth.",
        "sadness": "Reach out to a friend or try a short mindfulness meditation.",
        "anger": "Take deep breaths or write down your feelings before reacting.",
        "fear": "List your fears and challenge them with facts. Try grounding techniques.",
        "surprise": "Pause and process the surprise. Share your feelings with someone you trust.",
        "other": "Take a moment to check in with yourself."
    }
    suggestion = suggestions.get(emotions, suggestions["other"])
    state["suggestion"] = suggestion
    return state
