# selfcare_rag_suggester.py - Enhanced version
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os

def rag_selfcare_suggestion(state):
    """
    Enhanced RAG suggestion with better error handling and personalization
    """
    emotion = state.get("emotions", "").strip()
    user_input = state.get("text", "")
    
    # Validate input
    if not emotion:
        return {
            **state,
            "rag_self_care": "I'd like to give you personalized suggestions. Could you tell me more about how you're feeling?",
            "next_action": "wait_for_input",
            "expected_input": "emotion_clarification"
        }
    
    try:
        # Load RAG database
        embed_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        vectorstore = FAISS.load_local(
            "data/selfcare_rag",
            embed_model,
            allow_dangerous_deserialization=True
        )
        
        # Search for relevant content
        # Use both emotion and user input for better matching
        search_query = f"{emotion} {user_input}"
        docs = vectorstore.similarity_search(search_query, k=3)
        
        if not docs:
            # Fallback search with just emotion
            docs = vectorstore.similarity_search(emotion, k=3)
        
        if not docs:
            return {
                **state,
                "rag_self_care": "I don't have specific suggestions for this situation right now, but I'm here to support you.",
                "next_action": "continue"
            }
        
        # Extract content
        content = "\n".join([doc.page_content for doc in docs])
        
        # Generate personalized response
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        prompt = f"""
        Based on this self-care content:
        {content}
        
        User is feeling: {emotion}
        User context: {user_input}
        
        Provide personalized, actionable self-care suggestions that:
        1. Are specific to their emotional state
        2. Are practical and doable today
        3. Are empathetic and supportive
        4. Include both immediate relief and longer-term strategies
        
        Keep response under 200 words and focus on what they can do right now.
        """
        
        response = model.invoke(prompt)
        suggestion = response.content if hasattr(response, 'content') else str(response)
        
        return {
            **state,
            "rag_self_care": suggestion,
            "next_action": "continue"
        }
        
    except FileNotFoundError:
        return {
            **state,
            "rag_self_care": "I'm still learning about self-care strategies. Let me help you with some general techniques that work well.",
            "rag_error": "RAG database not found",
            "next_action": "continue"
        }
    except Exception as e:
        print(f"RAG suggestion failed: {e}")
        return {
            **state,
            "rag_self_care": "I'm having trouble accessing my knowledge base right now, but I'm here to help you through this.",
            "rag_error": str(e),
            "next_action": "continue"
        }

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