import streamlit as st
import requests

st.title("CalmBot - Mental Health Assistant")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

def get_agent_response(user_input):
    response = requests.post(
        "http://localhost:8000/analyze",
        json={"user_input": user_input}
    )
    return response.json()

user_input = st.text_input("How are you feeling today?", key="user_input")

if st.button("Send") and user_input:
    st.session_state.conversation.append(("You", user_input))
    result = get_agent_response(user_input)
    # Fallback: if agent_message is empty, use crisis_response if present
    agent_message = result.get("agent_message")
    if not agent_message and result.get("crisis_response"):
        agent_message = result["crisis_response"]
    st.session_state.conversation.append(("CalmBot", agent_message))

    # If clarification is needed, prompt for more input
    while result.get("needs_clarification"):
        clarification = st.text_input("CalmBot needs more info:", key=f"clarify_{len(st.session_state.conversation)}")
        if st.button("Send Clarification", key=f"clarify_btn_{len(st.session_state.conversation)}") and clarification:
            st.session_state.conversation.append(("You", clarification))
            result = get_agent_response(clarification)
            # Fallback for clarification as well
            agent_message = result.get("agent_message")
            if not agent_message and result.get("crisis_response"):
                agent_message = result["crisis_response"]
            st.session_state.conversation.append(("CalmBot", agent_message))
            st.json(result)  # Add this after result = get_agent_response(user_input)


for speaker, message in st.session_state.conversation:
    st.markdown(f"**{speaker}:** {message}")
