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
    st.session_state.conversation.append(("CalmBot", result["agent_message"]))

    # If clarification is needed, prompt for more input
    while result.get("needs_clarification"):
        clarification = st.text_input("CalmBot needs more info:", key=f"clarify_{len(st.session_state.conversation)}")
        if st.button("Send Clarification", key=f"clarify_btn_{len(st.session_state.conversation)}") and clarification:
            st.session_state.conversation.append(("You", clarification))
            result = get_agent_response(clarification)
            st.session_state.conversation.append(("CalmBot", result["agent_message"]))

for speaker, message in st.session_state.conversation:
    st.markdown(f"**{speaker}:** {message}")
