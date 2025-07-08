# self_care_recommender.py - Enhanced version
from langchain_google_genai import ChatGoogleGenerativeAI
import os

def suggest_care(state):
    """
    Enhanced self-care recommendation with personalization
    """
    emotions = state.get("emotions", "").lower()
    user_input = state.get("text", "")
    rag_suggestions = state.get("rag_self_care", "")
    web_articles = state.get("self_care_articles", [])
    
    # If no emotions detected, ask for clarification
    if not emotions or emotions == "neutral":
        return {
            **state,
            "suggestion": "I'd like to give you the best self-care suggestions. Could you tell me more about how you're feeling right now?",
            "next_action": "wait_for_input",
            "expected_input": "emotion_clarification"
        }
    
    # Basic suggestions mapping
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
    
    # Get basic suggestion
    primary_emotion = emotions.split(",")[0].strip() if "," in emotions else emotions
    suggestion = basic_suggestions.get(primary_emotion, basic_suggestions["other"])
    
    # If we have additional context, enhance the suggestion
    if rag_suggestions or web_articles:
        try:
            model = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                google_api_key=os.getenv("GEMINI_API_KEY")
            )
            
            context = ""
            if rag_suggestions:
                context += f"RAG suggestions: {rag_suggestions}\n"
            if web_articles:
                context += f"Web resources: {str(web_articles)[:500]}...\n"
            
            prompt = f"""
            Create a personalized self-care recommendation for someone feeling {emotions}.
            
            User context: {user_input}
            Basic suggestion: {suggestion}
            Additional context: {context}
            
            Provide:
            1. One immediate action they can take (2-5 minutes)
            2. One longer-term strategy (15-30 minutes)
            3. One resource or technique for ongoing support
            
            Keep it practical, empathetic, and actionable. Max 150 words.
            """
            
            enhanced_suggestion = model.invoke(prompt)
            suggestion = enhanced_suggestion.content if hasattr(enhanced_suggestion, 'content') else str(enhanced_suggestion)
            
        except Exception as e:
            # Fall back to basic suggestion if AI enhancement fails
            print(f"AI enhancement failed: {e}")
            pass
    
    return {
        **state,
        "suggestion": suggestion,
        "next_action": "continue"
    }