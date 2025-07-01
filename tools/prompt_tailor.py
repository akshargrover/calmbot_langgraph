def tailor_prompt(state):
    emotion = state.get("emotions", "other").lower()
    base_prompt = {
        "anxiety": "It's okay to feel anxious. Would you like to try a calming exercise?",
        "joy": "That's wonderful! Would you like to reflect on what brought you joy?",
        "shame": "Remember, everyone makes mistakes. Would you like to talk about it?",
        "gratitude": "Gratitude is powerful. Want to express it to someone?",
        "sadness": "It's okay to feel sad. Would you like some support or a self-care tip?",
        "anger": "Anger is a valid emotion. Would you like to try a grounding technique?",
        "fear": "Facing fears is brave. Would you like to explore this feeling?",
        "surprise": "Surprises can be good or bad. Want to share more?",
        "other": "How are you feeling right now? Would you like to talk more?"
    }
    tailored = base_prompt.get(emotion, base_prompt["other"])
    state["tailored_prompt"] = tailored
    return state
