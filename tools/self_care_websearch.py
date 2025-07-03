from langchain.tools.tavily_search import TavilySearchResults

tavily = TavilySearchResults(k=3)

def search_self_care_methods(state):
    emotion = state["emotions"]
    query = f"effective self-care strategies for dealing with {emotion}"
    results = tavily.run(query)
    return {**state, "self_care_articles": results}
