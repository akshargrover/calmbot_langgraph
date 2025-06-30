def generate_prompt(state):
    emotions = state["emotions"]
    forecast = state["forecast"]
    
    prompt = f"I sense {emotions}. Based on your recent trend: {forecast}. Would you like a reflection or a calming activity?"
    
    return {**state, "adaptive_prompt": prompt}
