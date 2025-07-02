def emotion_router(state):
    emotion = state.get("emotions", "other").lower()
    if emotion in ["anxiety", "fear", "shame"]:
        return "support_flow"
    elif emotion in ["joy", "gratitude", "surprise"]:
        return "positive_flow"
    else:
        return "default_flow"

