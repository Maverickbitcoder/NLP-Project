import streamlit as st
import pickle
import neattext.functions as nfx

# Load model and vectorizer
model = pickle.load(open("emotion_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

# Emotion emoji dictionary
emotion_emoji = {
    "anger":"😡",
    "empty":"😶",
    "enthusiasm":"🤩",
    "fun":"😄",
    "happiness":"😊",
    "hate":"💢",
    "love":"❤️",
    "neutral":"😐",
    "relief":"😌",
    "sadness":"😢",
    "surprise":"😲",
    "worry":"😟"
}

# Emotion colors
emotion_colors = {
    "anger":"#ff4d4d",
    "sadness":"#5dade2",
    "happiness":"#58d68d",
    "love":"#ff85c1",
    "surprise":"#f7dc6f",
    "neutral":"#d7dbdd"
}

# Page config
st.set_page_config(page_title="Emotion Detector", page_icon="😊", layout="centered")
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #e0f7fa, #fce4ec);
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
.stApp {
background: linear-gradient(135deg, #74ebd5 0%, #ACB6E5 100%);
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Emotion Detection from Text")
st.write("Type a sentence and the model will detect the emotion.")

# Session state for input
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

st.subheader("Try an Example")

col1, col2, col3 = st.columns(3)

if col1.button("😊 I feel amazing today"):
    st.session_state.user_input = "I feel amazing today"

if col2.button("😢 I am very sad"):
    st.session_state.user_input = "I am very sad"

if col3.button("😡 I hate this situation"):
    st.session_state.user_input = "I hate this situation"

user_input = st.text_area("Enter your text here", value=st.session_state.user_input)

# Cleaning function
def clean_text(text):
    text = nfx.remove_userhandles(text)
    text = nfx.remove_stopwords(text)
    text = nfx.remove_urls(text)
    text = nfx.remove_special_characters(text)
    return text.lower().strip()

# Prediction
if st.button("🔍 Detect Emotion"):

    if user_input.strip() == "":
        st.warning("Please enter some text")

    else:
        cleaned = clean_text(user_input)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]

        emoji = emotion_emoji.get(prediction,"")
        color = emotion_colors.get(prediction,"#d7dbdd")

        st.markdown("### 🎯 Prediction")

        st.markdown(
            f"""
            <div style="background-color:{color};
                        padding:20px;
                        border-radius:10px;
                        text-align:center;
                        font-size:24px;
                        font-weight:bold;">
                {emoji} Emotion Detected: {prediction.upper()}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("Cleaned text:", cleaned)