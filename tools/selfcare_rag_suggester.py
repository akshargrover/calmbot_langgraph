# selfcare_rag_suggester.py - Enhanced version
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os

# Additional helper function for emotion validation
def validate_emotion_input(state):
    """
    Helper function to validate if emotion input is sufficient
    """
    emotions = state.get("emotions", "").lower()
    
    # Check if emotions are too vague
    vague_emotions = ["neutral", "fine", "okay", "normal", "mixed", "confused"]
    
    if not emotions or emotions in vague_emotions:
        return False, "I'd like to understand your feelings better. Could you describe what you're experiencing in more detail?"
    
    # Check if emotions are just negations
    if emotions in ["not good", "not okay", "not fine", "bad"]:
        return False, "I hear that you're not feeling well. Can you tell me more specifically what emotions you're experiencing?"
    
    return True, ""

# Integration helper for the main graph
def create_self_care_chain(state):
    """
    Helper function to chain self-care nodes with proper validation
    """
    # Validate emotion input first
    is_valid, message = validate_emotion_input(state)
    
    if not is_valid:
        return {
            **state,
            "suggestion": message,
            "next_action": "wait_for_input",
            "expected_input": "emotion_clarification"
        }
    
    # Proceed with self-care chain
    return {
        **state,
        "next_action": "continue"
    }

def suggest_care(state):
    # --- Basic suggestion logic ---
    basic_suggestions = {
        "anxiety": "Try a 4-7-8 breathing exercise: breathe in for 4, hold for 7, exhale for 8.",
        "depression": "Consider a gentle walk outside, even just for 5 minutes, or reach out to someone you trust.",
        "joy": "Reflect on what brought you joy. Consider writing about it or sharing with someone.",
        "gratitude": "Write a short thank-you message to someone who has made a difference in your life.",
        "shame": "Practice self-compassion. Remember, everyone makes mistakes - they're part of growth.",
        "sadness": "Allow yourself to feel sad - it's valid. Try gentle movement or connecting with a friend.",
        "anger": "Take 5 deep breaths before reacting. Consider writing down your feelings first.",
        "fear": "Try grounding: name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste.",
        "stress": "Try progressive muscle relaxation: tense and release each muscle group for 5 seconds.",
        "loneliness": "Reach out to one person today, even with a simple 'thinking of you' message.",
        "grief": "Honor your feelings. Consider creating a small ritual or memory to acknowledge your loss.",
        "overwhelm": "Break down your tasks into smaller steps. Focus on just one thing at a time.",
        "other": "Take a moment to check in with yourself and acknowledge how you're feeling."
    }
    emotions = (state.get("emotions") or "").lower()
    primary_emotion = emotions.split(",")[0].strip() if "," in emotions else emotions
    basic = basic_suggestions.get(primary_emotion, basic_suggestions["other"])

    # --- RAG suggestion logic ---
    rag_suggestion = None
    try:
        embed_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        vectorstore = FAISS.load_local(
            "data/selfcare_rag",
            embed_model,
            allow_dangerous_deserialization=True
        )
        user_input = state.get("text", "")
        search_query = f"{emotions} {user_input}"
        docs = vectorstore.similarity_search(search_query, k=3)
        if not docs:
            docs = vectorstore.similarity_search(emotions, k=3)
        if docs:
            content = "\n".join([doc.page_content for doc in docs])
            model = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=os.getenv("GEMINI_API_KEY")
            )
            prompt = f"""
            Based on this self-care content:
            {content}
            User is feeling: {emotions}
            User context: {user_input}
            Provide personalized, actionable self-care suggestions that:
            1. Are specific to their emotional state
            2. Are practical and doable today
            3. Are empathetic and supportive
            4. Include both immediate relief and longer-term strategies
            Keep response under 200 words and focus on what they can do right now.
            """
            response = model.invoke(prompt)
            rag_suggestion = response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        print(f"Unified suggest_care: RAG suggestion failed: {e}")
        rag_suggestion = None

    # --- Combine and return ---
    if rag_suggestion:
        combined = f"Basic self-care tip: {basic}\n\nPersonalized suggestion: {rag_suggestion}"
    else:
        combined = f"Basic self-care tip: {basic}"
    return {
        **state,
        "suggestion": combined,
        "agent_output": combined,
        "next_action": "continue"
    }