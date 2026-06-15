import streamlit as st

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰"
)

st.title("📰 AI-Based Fake News Detection System")

st.markdown("""
### Welcome!

This application uses a Deep Learning (LSTM) model to classify news articles as **Real News** or **Fake News**.

📌 Enter a news article below and click **Predict**.
""")

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

    if score > 0.5:
        st.success("🟢 Real News")
    else:
        st.error("🔴 Fake News")

st.markdown("---")
st.caption("Developed by Meghasree | AI-Based Fake News Detection System")