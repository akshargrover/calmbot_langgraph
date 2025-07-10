import streamlit as st
import requests

st.set_page_config(page_title="CalmBot - Mental Health Assistant", page_icon="🧘")
st.title("🧘 CalmBot - Mental Health Assistant")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Chat message avatars
USER_AVATAR = "🧑"
BOT_AVATAR = "🤖"

# Chat message container
chat_container = st.container()

# Input at the bottom
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message...", key="user_input", placeholder="How are you feeling today?")
    send_clicked = st.form_submit_button("Send")

if send_clicked and user_input:
    st.session_state.conversation.append(("user", user_input))
    result = requests.post(
        "http://localhost:8000/analyze",
        json={"user_input": user_input}
    ).json()
    # Fallback: if agent_message is empty, use crisis_response if present
    agent_message = result.get("agent_message")
    if not agent_message and result.get("crisis_response"):
        agent_message = result["crisis_response"]
    st.session_state.conversation.append(("bot", agent_message))

    # If clarification is needed, prompt for more input
    while result.get("needs_clarification"):
        clarification = st.text_input("CalmBot needs more info:", key=f"clarify_{len(st.session_state.conversation)}")
        if st.button("Send Clarification", key=f"clarify_btn_{len(st.session_state.conversation)}") and clarification:
            st.session_state.conversation.append(("user", clarification))
            result = requests.post(
                "http://localhost:8000/analyze",
                json={"user_input": clarification}
            ).json()
            agent_message = result.get("agent_message")
            if not agent_message and result.get("crisis_response"):
                agent_message = result["crisis_response"]
            st.session_state.conversation.append(("bot", agent_message))
            st.json(result)

# Display chat history in a modern chat style
with chat_container:
    for speaker, message in st.session_state.conversation:
        if speaker == "user":
            st.markdown(
                f"<div style='display: flex; align-items: center; margin-bottom: 8px;'>"
                f"<span style='font-size: 1.5em; margin-right: 8px;'>{USER_AVATAR}</span>"
                f"<div style='background: #388E3C; color: #fff; padding: 10px 16px; border-radius: 16px; max-width: 70%; margin-left: 4px;'>"
                f"{message}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='display: flex; align-items: center; margin-bottom: 8px; justify-content: flex-end;'>"
                f"<div style='background: #333333; color: #fff; padding: 10px 16px; border-radius: 16px; max-width: 70%; margin-right: 4px;'>"
                f"{message}</div><span style='font-size: 1.5em; margin-left: 8px;'>{BOT_AVATAR}</span></div>", unsafe_allow_html=True)

# Auto-scroll to the latest message (Streamlit does this by default, but this ensures it)
# st.experimental_rerun()
