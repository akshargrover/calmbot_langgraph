from langchain_tavily import TavilySearch

tavily = TavilySearch(k=3)

def search_emotion_context(state):
    emotion = state["emotions"]
    query = f"recent news or stories about {emotion} and mental health"
    results = tavily.run(query)
    return {**state, "emotion_context_links": results}
