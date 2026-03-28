"""
Simple Streamlit Web Interface for NLP_Recognition Chatbot
Run with: streamlit run simple_streamlit_app.py
"""

import streamlit as st
from chatbot import NLPChatbot
from knowledge_base import KNOWLEDGE_BASE

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="NLP_Recognition Chatbot",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==================== SESSION STATE ====================
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = NLPChatbot()

if 'messages' not in st.session_state:
    st.session_state.messages = []

# ==================== HEADER ====================
st.markdown("""
    <style>
    .main { max-width: 900px; }
    </style>
""", unsafe_allow_html=True)

st.title(" NLP_Recognition Chatbot")
st.write("Ask me about NLP_Recognition concepts and AI applications!")
st.divider()

# ==================== CHAT AREA ====================
chat_container = st.container()

with chat_container:
    for idx, msg in enumerate(st.session_state.messages):
        if msg['role'] == 'user':
            with st.chat_message("user"):
                st.write(msg['content'])
        else:
            with st.chat_message("assistant"):
                st.write(msg['content'])
                if msg.get('intent'):
                    st.caption(f"✓ Intent: {msg['intent']} | Confidence: {msg['confidence']:.0%}")
                else:
                    st.caption("✗ Intent: Not recognized")

# ==================== USER INPUT ====================
st.divider()
user_input = st.chat_input("Type your question here...")

if user_input:
    # Add user message to display
    st.session_state.messages.append({
        'role': 'user',
        'content': user_input
    })

    # Get bot response
    response, intent, confidence = st.session_state.chatbot.get_response(user_input)

    # Add bot message to display
    st.session_state.messages.append({
        'role': 'assistant',
        'content': response,
        'intent': intent,
        'confidence': confidence
    })

    # Rerun to show new messages
    st.rerun()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title(" Menu")

    # Quick Questions
    st.subheader(" Quick Questions")

    quick_questions = [
        "What is tokenization?",
        "Explain sentiment analysis",
        "What are chatbots?",
        "Tell me about machine learning",
        "What is NLP_Recognition?"
    ]

    for q in quick_questions:
        if st.button(q, use_container_width=True):
            # Add to messages
            st.session_state.messages.append({
                'role': 'user',
                'content': q
            })

            # Get response
            response, intent, confidence = st.session_state.chatbot.get_response(q)

            # Add response to messages
            st.session_state.messages.append({
                'role': 'assistant',
                'content': response,
                'intent': intent,
                'confidence': confidence
            })

            st.rerun()

    st.divider()

    # Statistics
    st.subheader(" Stats")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Messages", len(st.session_state.messages))

    with col2:
        st.metric("Topics", len(KNOWLEDGE_BASE) - 2)

    st.divider()

    # Actions
    st.subheader(" Actions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button(" Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chatbot = NLPChatbot()
            st.rerun()

    with col2:
        if st.button(" Reset", use_container_width=True):
            st.session_state.chatbot = NLPChatbot()
            st.info("Bot reset!")

    st.divider()

    # Available Topics
    st.subheader("Topics")

    topics = []
    for intent in KNOWLEDGE_BASE.keys():
        if intent not in ['greeting', 'goodbye']:
            topics.append(intent.replace('_', ' ').title())

    for i, topic in enumerate(sorted(topics)):
        st.caption(f"{i + 1}. {topic}")

# ==================== FOOTER ====================
st.divider()
st.caption(" NLP_Recognition Chatbot v1.0 | Ask away! ")