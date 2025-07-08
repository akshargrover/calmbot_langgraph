# self_care_websearch.py - Enhanced version
from langchain_tavily import TavilySearch

def search_self_care_methods(state):
    """
    Enhanced web search with better error handling and validation
    """
    emotion = state.get("emotions", "").strip()
    
    # Validate input
    if not emotion:
        return {
            **state,
            "self_care_articles": [],
            "next_action": "continue"
        }
    
    try:
        tavily = TavilySearch(k=3)
        
        # Create more specific queries based on emotion
        if emotion.lower() in ["anxiety", "panic", "worry"]:
            query = f"evidence-based anxiety management techniques and coping strategies"
        elif emotion.lower() in ["depression", "sadness", "hopelessness"]:
            query = f"depression self-care strategies mental health support"
        elif emotion.lower() in ["stress", "overwhelm"]:
            query = f"stress management techniques mindfulness relaxation"
        elif emotion.lower() in ["anger", "frustration", "irritation"]:
            query = f"anger management techniques healthy expression"
        else:
            query = f"effective self-care strategies for {emotion} mental health"
        
        results = tavily.run(query)
        
        # Filter and validate results
        if isinstance(results, list):
            # Keep only relevant results
            filtered_results = []
            for result in results[:3]:  # Limit to top 3
                if isinstance(result, dict):
                    filtered_results.append(result)
                elif isinstance(result, str):
                    filtered_results.append({"content": result})
            results = filtered_results
        
        return {
            **state,
            "self_care_articles": results,
            "next_action": "continue"
        }
        
    except Exception as e:
        print(f"Web search failed: {e}")
        return {
            **state,
            "self_care_articles": [],
            "web_search_error": str(e),
            "next_action": "continue"
        }