import streamlit as st
import requests

st.set_page_config(page_title="CalmBot - Mental Health Assistant", page_icon="🧘")
st.title("🧘 CalmBot - Mental Health Assistant")
API_URL = "http://localhost:8000"

USER_AVATAR = "🧑"
BOT_AVATAR = "🤖"

if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "pending_user_input" not in st.session_state:
    st.session_state.pending_user_input = None
if "last_backend_response" not in st.session_state:
    st.session_state.last_backend_response = {}

# Clear Memory Button
if st.button("Clear Memory / Reset Conversation"):
    response = requests.post(f"{API_URL}/clear_memory", json={"user_id": "demo_user"})
    if response.status_code == 200 and response.json().get("success"):
        st.success("Memory cleared! Start a new conversation.")
        st.session_state.conversation = []
    else:
        st.error("Failed to clear memory.")

# Chat display
chat_container = st.container()
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

# Determine the prompt for the input box
prompt = "Type your message..."
if st.session_state.last_backend_response.get("waiting_for_input"):
    # Use the backend's prompt if available
    prompt = st.session_state.last_backend_response.get("agent_message", prompt)

# Input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input(prompt, key="user_input", placeholder="How are you feeling today?")
    send_clicked = st.form_submit_button("Send")

if send_clicked and user_input:
    st.session_state.pending_user_input = user_input

# Process pending user input and get bot response
if st.session_state.pending_user_input:
    st.session_state.conversation.append(("user", st.session_state.pending_user_input))
    try:
        response = requests.post(f"{API_URL}/analyze", json={"user_input": st.session_state.pending_user_input})
        if response.status_code == 200:
            result = response.json()
        else:
            st.error("Backend error: " + response.text)
            result = {"agent_message": "Sorry, something went wrong."}
    except Exception as e:
        st.error(f"Request failed: {e}")
        result = {"agent_message": "Sorry, something went wrong."}
    agent_message = result.get("agent_message")
    if not agent_message and result.get("crisis_response"):
        agent_message = result["crisis_response"]
    st.session_state.conversation.append(("bot", agent_message))
    st.session_state.last_backend_response = result  # Store the full backend response
    st.session_state.pending_user_input = None
    st.rerun()