import streamlit as st

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰"
)

st.title("📰 AI-Based Fake News Detection System")
with st.sidebar:
    st.header("About")
    st.write("Fake News Detection using Machine Learning")
    st.write("Built using Python, TensorFlow and Streamlit")



st.markdown("""
<style>
.stApp {
    background-color: #F8FAFC;
    color: black;
}

h1, h2, h3, h4, h5, h6 {
    color: #111827;
}

section[data-testid="stSidebar"] {
    background-color: #161B22;
}

div.stButton > button {
    background-color: #1E88E5;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #1565C0;
}

textarea {
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

news_text = st.text_area(
    "Enter News Article",
    height=200
)


import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved files
lstm_model = load_model("notebooks/best_lstm_model.keras")

with open("notebooks/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

if st.button("Predict"):

    if not news_text.strip():
        st.warning("⚠️ Please enter a news article.")
        st.stop()

    seq = tokenizer.texts_to_sequences([news_text])
    padded = pad_sequences(seq, maxlen=200)

    with st.spinner("Analyzing news article..."):
        prediction = lstm_model.predict(padded)

    score = float(prediction[0][0])

    st.subheader("Prediction Result")

    st.write(f"Confidence Score: {score*100:.2f}%")
    st.progress(score)

    if score > 0.5:
        st.success("🟢 This article is predicted as REAL NEWS")
    else:
        st.error("🔴 This article is predicted as FAKE NEWS")

st.markdown("---")
st.caption("Developed by Meghasree | AI-Based Fake News Detection System")