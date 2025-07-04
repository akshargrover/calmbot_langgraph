from langchain.agents import AgentExecutor, initialize_agent, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
import json
from config.settings import get_gemini_api_key

# ------------------ TOOL WRAPPERS ------------------

@tool
def emotion_context_search_tool(query: str) -> str:
    """Get real-time emotion context from the web."""
    from tools.emotion_context_search import search_emotion_context
    result = search_emotion_context({"emotions": query})["emotion_context_links"]
    return json.dumps(result)

@tool
def self_care_websearch_tool(query: str) -> str:
    """Get real-time self-care strategies for an emotion."""
    from tools.self_care_websearch import search_self_care_methods
    result = search_self_care_methods({"emotions": query})["self_care_articles"]
    return json.dumps(result)

@tool
def rag_self_care_tool(query: str) -> str:
    """Retrieve self-care suggestions from internal RAG database."""
    from tools.selfcare_rag_suggester import rag_selfcare_suggestion
    result = rag_selfcare_suggestion({"emotions": query})["rag_self_care"]
    return result

def crisis_router(state):
    emotion = state.get("emotions", "").lower()
    suicidal_keywords = ["suicidal", "want to die", "end my life", "no will to live", "kill myself"]
    if any(keyword in emotion for keyword in suicidal_keywords):
        return "crisis"
    return "normal"

@tool

# ------------------ AGENT NODE ------------------

def agent_router_node(state):
    emotion = state.get("emotions", "")
    previous_trace = state.get("router_trace", [])

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=get_gemini_api_key())
    tools = [emotion_context_search_tool, self_care_websearch_tool, rag_self_care_tool]

    agent = initialize_agent(
        tools=tools,
        llm=model,
        agent_type="openai-functions",
        verbose=True
    )

    prompt = f"""
    You are a mental health assistant router. Based on the user's detected emotion '{emotion}', choose the most helpful tools. 
    - If the emotion is intense (e.g., anxiety, grief), use all tools.
    - If it's positive (e.g., joy, gratitude), just fetch a self-care tip.
    - Return your decisions in natural language.
    - If the user is in crisis, return "crisis"
    - If the user is not in crisis, return "normal"

    """

    # Run the agent
    output = agent.run(prompt)

    # Add to decision history
    current_trace = {
        "emotion": emotion,
        "prompt": prompt,
        "tools_run": [tool.name for tool in tools],
        "agent_summary": output
    }

    full_trace = previous_trace + [current_trace]

    return {
        **state,
        "agent_router_output": output,
        "router_trace": full_trace  # stores all router decisions
    }
