import streamlit as st

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰"
)

st.title("📰 Fake News Detection using AI/ML")
st.write("Enter a news article below and check whether it is Real or Fake.")

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
st.write("App loaded successfully")

if st.button("Predict"):

    seq = tokenizer.texts_to_sequences([news_text])
    padded = pad_sequences(seq, maxlen=200)

    prediction = lstm_model.predict(padded)

    st.write(f"Confidence Score: {prediction[0][0]:.4f}")

    if prediction[0][0] > 0.5:
        st.success("🟢 Real News")
    else:
        st.error("🔴 Fake News")