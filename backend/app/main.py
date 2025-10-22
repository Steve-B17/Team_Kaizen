# chat_ui_intent_feedback.py
import streamlit as st

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "awaiting_feedback" not in st.session_state:
    st.session_state.awaiting_feedback = False

st.set_page_config(page_title="Intent Classification Chat", page_icon="🤖")
st.title("🤖 Intent Classification Chat UI")

# Function to add a message
def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    elif msg["role"] == "bot":
        st.markdown(f"**Model:** {msg['content']}")
    elif msg["role"] == "feedback":
        st.markdown(f"**Feedback:** {msg['content']}")

# User input
if not st.session_state.awaiting_feedback:
    user_input = st.text_input("Enter your utterance:", key="utterance_input")

    if user_input:
        add_message("user", user_input)

        # TODO: Replace this with your actual intent classification model
        model_response = f"Predicted Request Type: **BOOK_FLIGHT**"  # Example
        add_message("bot", model_response)

        st.session_state.awaiting_feedback = True
        st.experimental_rerun()
else:
    # Feedback buttons
    st.markdown("Was the prediction correct?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes"):
            add_message("feedback", "Yes")
            st.session_state.awaiting_feedback = False
            st.experimental_rerun()
    with col2:
        if st.button("❌ No"):
            add_message("feedback", "No")
            st.session_state.awaiting_feedback = False
            st.experimental_rerun()
